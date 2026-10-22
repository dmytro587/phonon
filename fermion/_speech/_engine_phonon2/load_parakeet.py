"""load_parakeet(): Phonon-2 container -> mlx-audio ParakeetTDT, packed (default) or dense.

  container  : fermion-five-value-parakeet-v1 (five-value planes + int6/int8 tables + fp16)
  config     : mlx-community NeMo-style config.json (vocabulary included) -- the layout
               mlx_audio.stt.models.parakeet.Model.from_config expects
  packed=True: the 264 quantised encoder linears become PackedFiveValueLinear (two native
               2-bit QMMs, per-row lo / hi metadata); everything else is dense in `dtype`
  packed=False: the same graph with the five-value weights expanded to dense `dtype`
               (fp16 is exact; this is the reference path for the exactness gate)

Shared pieces written once by the Detta C3 lane and vendored here with shas in
SHARED_FROM_DETTA_C3_LANE.sha256: fermion_container.read_container (the writer read
backwards) and hf_to_mlx_parakeet.hf_to_mlx (697-key name map + conv/LSTM layout).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fermion_container import read_container                      # noqa: E402
from hf_to_mlx_parakeet import check_against_reference, hf_to_mlx  # noqa: E402
from packed_runtime import (FiveValueRecord, five_value_hf_to_mlx,  # noqa: E402
                            plane_gate, substitute_five_value)
from mlx.utils import tree_flatten  # noqa: E402

DEFAULT_CONFIG = HERE / "mlx_config_parakeet_tdt_0.6b_v3.json"
REF_HEADER = HERE / "mlx_ref_header_ed2b7e8c.json"


def _install_fp32_frontend() -> None:
    """The log-mel front end is computed in fp32 and cast to the model dtype.

    mlx-audio computes `log_mel_spectrogram` in the AUDIO dtype. Its per-feature
    normalisation sums over the time axis, and in float16 that sum overflows at
    about 3,000 frames (30 s of audio): 60 s of speech gives a NaN mel and an
    empty transcript (measured 2026-09-24, receipts/nan_site_probe.json). Doing
    the front end in fp32 costs nothing measurable and removes the failure for
    every dtype; the encoder still runs in the caller's dtype.
    """
    from mlx_audio.stt.models.parakeet import parakeet as _pk
    from mlx_audio.stt.models.parakeet import audio as _audio
    if getattr(_pk, "_fermion_fp32_frontend", False):
        return
    _orig = _audio.log_mel_spectrogram

    def log_mel_spectrogram_fp32(x, args):
        return _orig(x.astype(mx.float32), args).astype(x.dtype)

    _pk.log_mel_spectrogram = log_mel_spectrogram_fp32
    _pk._fermion_fp32_frontend = True


def five_value_records(raw: dict) -> list[FiveValueRecord]:
    recs = []
    for name, r in raw.items():
        if "sign" in r:
            recs.append(FiveValueRecord(name, r["sign"], r["is_hi"], r["lo"], r["hi"]))
    return recs


def load_parakeet(container: str | Path, config: str | Path = DEFAULT_CONFIG, *,
                  dtype: mx.Dtype = mx.float16, packed: bool = True,
                  scale_dtype: mx.Dtype = mx.float32, receipts: dict | None = None):
    from mlx_audio.stt.models.parakeet import Model

    _install_fp32_frontend()
    t0 = time.perf_counter()
    tensors, index, raw = read_container(str(container), with_raw=True)
    t_read = time.perf_counter() - t0
    fv_names = [e["n"] for e in index if e["k"] == "five_value"]
    if packed:
        # Lean path: the fp16 expansion of the 264 five-value modules is never sent to
        # MLX; the planes are built from the raw records and the numpy copies freed as
        # we go (a 16 GB Mac swaps otherwise).
        for n in fv_names:
            del tensors[n + ".weight"]
    mlx_named = hf_to_mlx(tensors if not packed else {**tensors, **{n + ".weight": np.zeros(
        tuple(next(e["shape"] for e in index if e["n"] == n)), np.float16) for n in fv_names}})
    chk = None
    if REF_HEADER.exists():
        chk = check_against_reference(mlx_named, json.load(open(REF_HEADER)))
        if not chk["pass"]:
            raise RuntimeError(f"name map does not match the mlx-community reference: {chk}")
    cfg = json.load(open(config))
    model = Model.from_config(cfg)
    n_sub = 0
    plane = None
    if packed:
        fv_mlx = {five_value_hf_to_mlx(n) + ".weight" for n in fv_names}
        weights = [(k, mx.array(v).astype(dtype)) for k, v in mlx_named.items() if k not in fv_mlx]
        model.load_weights(weights, strict=False)
        # G-LOAD, explicit: every model parameter except the 264 five-value weights
        # was supplied, and nothing else was.
        want = {k for k, _ in tree_flatten(model.parameters())} - fv_mlx
        got = {k for k, _ in weights}
        if want != got:
            raise RuntimeError(f"G-LOAD: missing {sorted(want - got)[:5]} unexpected {sorted(got - want)[:5]}")
        del weights, tensors, mlx_named
        recs = []
        for n in fv_names:
            r = raw.pop(n)
            recs.append(FiveValueRecord(n, r["sign"], r["is_hi"], r["lo"], r["hi"]))
        sub = substitute_five_value(model, recs, scale_dtype=scale_dtype, expect_modules=264)
        n_sub = len(sub)
        if receipts is not None and receipts.get("want_plane_gate"):
            plane = plane_gate(model, recs)
        del recs
    else:
        weights = [(k, mx.array(v).astype(dtype)) for k, v in mlx_named.items()]
        model.load_weights(weights, strict=True)      # G-LOAD: strict -> 0 missing / 0 unexpected
        del weights, tensors, mlx_named
    del raw
    mx.eval(model.parameters())
    model.eval()
    t_all = time.perf_counter() - t0
    if receipts is not None:
        receipts.update({
            "container": str(container), "container_bytes": Path(container).stat().st_size,
            "records": len(index), "packed": packed, "substituted": n_sub,
            "dtype": str(dtype), "scale_dtype": str(scale_dtype),
            "read_s": round(t_read, 3), "load_s": round(t_all, 3),
            "name_map_check": chk,
            "plane_gate": None if plane is None else {k: v for k, v in plane.items() if k != "per_module"},
            "plane_gate_per_module_worst": None if plane is None else sorted(
                plane["per_module"], key=lambda r: -r["max_abs_delta"])[:3],
        })
    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("container")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--scale-dtype", default="float32")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--receipt", default=None)
    a = ap.parse_args()
    rc = {"want_plane_gate": True}
    m = load_parakeet(a.container, dtype=getattr(mx, a.dtype), packed=not a.dense,
                      scale_dtype=getattr(mx, a.scale_dtype), receipts=rc)
    rc["peak_memory_after_load_gb"] = mx.get_peak_memory() / 1e9
    rc["active_memory_after_load_gb"] = mx.get_active_memory() / 1e9
    if a.audio:
        t = time.perf_counter()
        r = m.generate(a.audio, dtype=getattr(mx, a.dtype))
        rc["text"] = r.text
        rc["decode_s"] = round(time.perf_counter() - t, 3)
    print(json.dumps(rc, indent=1, default=str))
    if a.receipt:
        json.dump(rc, open(a.receipt, "w"), indent=1, default=str)
