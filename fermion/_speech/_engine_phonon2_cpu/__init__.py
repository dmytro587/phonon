"""Phonon-2 packed five-value CPU GEMM: ctypes wrapper + torch module swap for the tier-1 CPU engine.

  lib = load_library(path)                     -> ctypes handle (abi check)
  PackedFiveValueCPU(rec, bias, lib)          -> nn.Module replacing an encoder nn.Linear / kernel-1 nn.Conv1d
  swap_encoder(model, raw_records, lib)       -> replaces the 264 five-value modules in a stock ParakeetForTDT
  unit tests: exactness (one-hot probes read every weight back exactly; int8 path vs fp32 reference of the same
  quantised activations equal to fp32-accumulation order), microbench vs torch fp32.
"""
from __future__ import annotations

import ctypes
import os
import platform
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _platform_name(prefix: str, base: bool = False) -> str | None:
    """File name of a kernel binary for this platform; `base` = the baseline-ISA variant (0.2.5: `-base` before the extension:
    armv8.0 NEON without dotprod on aarch64, no AVX2 anywhere on x86-64).  0.2.8: macOS x86_64 (Intel Macs), same shape as Linux x86-64."""
    sysname, mach = platform.system(), platform.machine().lower(); suf = "-base" if base else ""
    return {("Darwin", "arm64"): f"lib{prefix}-macos-arm64{suf}.dylib", ("Darwin", "x86_64"): f"lib{prefix}-macos-x86_64{suf}.dylib",
            ("Linux", "x86_64"): f"lib{prefix}-linux-x86_64{suf}.so", ("Linux", "aarch64"): f"lib{prefix}-linux-aarch64{suf}.so",
            ("Windows", "amd64"): f"{prefix}-windows-x64{suf}.dll"}.get((sysname, mach))


#: architectures whose PLAIN GEMM / TDT binaries assume more than the baseline ISA (built -march=armv8.2-a+dotprod), so the
#: baseline tier needs their -base variants too.  The x86-64 plain GEMM and TDT binaries are built for baseline x86-64 and
#: dispatch at run time; only the x86 ENCODER binary is AVX2-only.
_BASE_NEEDS_TDT = {"aarch64"}

_PLAN = None


def _plan() -> dict:
    """The CPU-feature plan (fermion._speech._cpu_features.plan), computed once per process BEFORE any dlopen."""
    global _PLAN
    if _PLAN is None:
        from fermion._speech import _cpu_features
        _PLAN = _cpu_features.plan()
    return _PLAN


def default_library() -> Path:
    """The packed GEMM library for THIS processor's tier.  `libphonon2_enc-*` (0007b) is a superset — same GEMM symbols + the
    C encoder — and is preferred when it ships beside the plain library.  0.2.5: the tier comes from runtime CPU-feature
    detection (dotprod / i8mm on aarch64, AVX2 and up on x86-64); processors without the native tier's ISA get the -base
    binaries, and a missing -base binary raises (the engine then takes the full-precision path) instead of loading a binary
    that would fault."""
    override = os.environ.get("PHONON2_CPU_LIBRARY")
    if override:
        return Path(override)
    p = _plan()
    if p["refusal"]:
        raise RuntimeError(p["refusal"])
    base = p["base"]; pins = _pins()
    enc = _platform_name("phonon2_enc", base=base)
    if enc and (HERE / enc).is_file() and enc in pins and (base or os.environ.get("FERMION_P2_CPU_ENC", "c").lower() == "c"):
        return HERE / enc                            # on the baseline tier the -base encoder library is the only GEMM carrier
    if base:
        from fermion._speech import _cpu_features
        raise RuntimeError(_cpu_features.missing_line(p["detect"], f"baseline kernel binary ({enc})"))
    name = _platform_name("phonon2_cpu")
    return HERE / (name or "libphonon2_cpu.so")


def _pins() -> dict:
    """SHA256SUMS.txt beside the binaries: `<sha256>  <filename>` per line (written by ship/build_gate.sh)."""
    f = HERE / "SHA256SUMS.txt"
    if not f.is_file():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def verified_library(path=None) -> Path:
    """The platform binary, refused if its sha256 does not match the shipped pin (an override path is trusted as-is)."""
    import hashlib
    p = Path(path or default_library())
    if os.environ.get("PHONON2_CPU_LIBRARY") or path is not None:
        return p
    if not p.is_file():
        raise FileNotFoundError(f"Phonon-2 CPU kernel binary for this platform is not in the package: {p.name}")
    pin = _pins().get(p.name)
    if pin is None:
        raise RuntimeError(f"{p.name}: no sha256 pin shipped for this binary")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != pin:
        raise RuntimeError(f"{p.name}: sha256 {h[:16]}… does not match the shipped pin {pin[:16]}…; refusing to load")
    return p


def load_library(path=None):
    lib = ctypes.CDLL(str(verified_library(path)))
    lib.phonon2_cpu_abi_version.restype = ctypes.c_int
    assert lib.phonon2_cpu_abi_version() == 1, "ABI mismatch"
    lib.phonon2_cpu_create.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    lib.phonon2_cpu_create.restype = ctypes.c_void_p
    lib.phonon2_cpu_destroy.argtypes = (ctypes.c_void_p,)
    lib.phonon2_cpu_gemm.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    lib.phonon2_cpu_gemm.restype = ctypes.c_int
    lib.phonon2_cpu_dequant_row.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
    lib.phonon2_cpu_set_threads.argtypes = (ctypes.c_int,)
    lib.phonon2_cpu_set_kernel.argtypes = (ctypes.c_int,)
    lib.phonon2_cpu_kernel_in_use.restype = ctypes.c_int
    lib.phonon2_cpu_enable_onedot.argtypes = (ctypes.c_void_p,); lib.phonon2_cpu_enable_onedot.restype = ctypes.c_float
    lib.phonon2_cpu_set_onedot.argtypes = (ctypes.c_void_p, ctypes.c_int)
    if hasattr(lib, "phonon2_cpu_build_features"):  # 0.2.5 binaries: the C side's own host check, mirrored here before any create
        lib.phonon2_cpu_features.restype = ctypes.c_uint; lib.phonon2_cpu_build_features.restype = ctypes.c_uint
        need, have = lib.phonon2_cpu_build_features(), lib.phonon2_cpu_features()
        if need & ~have:
            raise RuntimeError(f"{Path(lib._name).name} assumes CPU features 0x{need:x}; this processor provides 0x{have:x}; not loading it")
    if hasattr(lib, "phonon2_enc_abi_version"):      # 0007b superset library
        lib.phonon2_enc_abi_version.restype = ctypes.c_int
        lib.phonon2_enc_create.restype = ctypes.c_void_p; lib.phonon2_enc_create.argtypes = [ctypes.c_int] * 7
        lib.phonon2_enc_set_subsampling.restype = ctypes.c_int; lib.phonon2_enc_set_subsampling.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 13
        lib.phonon2_enc_set_layer.restype = ctypes.c_int; lib.phonon2_enc_set_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        lib.phonon2_enc_out_len.restype = ctypes.c_int; lib.phonon2_enc_out_len.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.phonon2_enc_forward.restype = ctypes.c_int; lib.phonon2_enc_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.phonon2_enc_destroy.argtypes = [ctypes.c_void_p]
    return lib


class CEncoder:
    """The whole FastConformer encoder in C (phonon2_cpu_enc.c, 0007b): subsampling, relative positions, 24 blocks, on the
    packed one-dot GEMM handles the torch modules already hold and the library's single thread pool.  Same arithmetic class
    as the torch one-dot path (int8 activations at every linear, fp32 elsewhere); gated on the 2,939-clip test-other set."""

    def __init__(self, lib, model):
        if not hasattr(lib, "phonon2_enc_abi_version") or lib.phonon2_enc_abi_version() != 1:
            raise RuntimeError("the loaded library has no C encoder (or a different ABI)")
        enc = model.encoder; cfg = enc.config; self.lib = lib; self._keep = []
        d, H = cfg.hidden_size, cfg.num_attention_heads; self.d = d
        def F(t):
            a = np.ascontiguousarray(t.detach().float().numpy().reshape(-1), dtype=np.float32); self._keep.append(a); return a.ctypes.data
        self.h = lib.phonon2_enc_create(cfg.num_hidden_layers, d, cfg.intermediate_size, H, cfg.conv_kernel_size, cfg.num_mel_bins, cfg.subsampling_conv_channels)
        assert self.h, "phonon2_enc_create failed"
        sub = enc.subsampling.layers
        rc = lib.phonon2_enc_set_subsampling(self.h, F(sub[0].weight), F(sub[0].bias), F(sub[2].weight), F(sub[2].bias), F(sub[3].weight), F(sub[3].bias),
                                             F(sub[5].weight), F(sub[5].bias), F(sub[6].weight), F(sub[6].bias), F(enc.subsampling.linear.weight),
                                             F(enc.subsampling.linear.bias), F(enc.encode_positions.inv_freq))
        if rc != 0:
            raise RuntimeError(f"phonon2_enc_set_subsampling rc={rc}")
        for l, blk in enumerate(enc.layers):
            a = blk.self_attn; c = blk.conv
            mods = [blk.feed_forward1.linear1, blk.feed_forward1.linear2, a.q_proj, a.k_proj, a.v_proj, a.o_proj, a.relative_k_proj,
                    c.pointwise_conv1, c.pointwise_conv2, blk.feed_forward2.linear1, blk.feed_forward2.linear2]
            if not all(hasattr(mm, "pm") for mm in mods):
                raise RuntimeError("C encoder needs the packed encoder modules (FERMION_P2_CPU=onedot/packed)")
            vecs = [blk.norm_feed_forward1.weight, blk.norm_feed_forward1.bias, blk.norm_self_att.weight, blk.norm_self_att.bias, blk.norm_conv.weight,
                    blk.norm_conv.bias, blk.norm_feed_forward2.weight, blk.norm_feed_forward2.bias, blk.norm_out.weight, blk.norm_out.bias, a.bias_u, a.bias_v,
                    c.depthwise_conv.weight, c.norm.weight, c.norm.bias, c.norm.running_mean, c.norm.running_var]
            M = (ctypes.c_void_p * 11)(*[mm.pm.h for mm in mods]); V = (ctypes.c_void_p * 17)(*[F(v) for v in vecs])
            if lib.phonon2_enc_set_layer(self.h, l, M, V) != 0:
                raise RuntimeError(f"phonon2_enc_set_layer({l}) failed")
        self._keep = self._keep[:0]                  # the library copied every vector

    def forward(self, feats: np.ndarray) -> np.ndarray:
        """feats [T][n_mels] float32 log-mel of ONE utterance -> encoder output [T8][d] float32."""
        x = np.ascontiguousarray(feats, dtype=np.float32); T = x.shape[0]; T8 = self.lib.phonon2_enc_out_len(self.h, T)
        out = np.empty((T8, self.d), dtype=np.float32)
        rc = self.lib.phonon2_enc_forward(self.h, x.ctypes.data, T, out.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"phonon2_enc_forward rc={rc}")
        return out

    def __del__(self):
        try:
            self.lib.phonon2_enc_destroy(self.h)
        except Exception:
            pass


def pack_planes(rec):
    """five-value record (sign int8, is_hi bool, lo/hi fp16) -> (plane_a, plane_b) uint8 [O][I/4] in the MLX byte layout."""
    sign = rec["sign"].astype(np.int8); is_hi = rec["is_hi"]
    qa = (sign + 1).astype(np.uint8); qb = (1 + sign * is_hi.astype(np.int8)).astype(np.uint8)
    def pack(q):
        o, i = q.shape; q = q.reshape(o, i // 4, 4)
        return (q[:, :, 0] | (q[:, :, 1] << 2) | (q[:, :, 2] << 4) | (q[:, :, 3] << 6)).astype(np.uint8)
    return np.ascontiguousarray(pack(qa)), np.ascontiguousarray(pack(qb))


class PackedMatrix:
    def __init__(self, lib, rec, planes=None):
        """`rec` = five-value record (sign/is_hi/lo/hi) -> planes packed here; or `planes` = (rows, cols, pa, pb, lo, hi)
        straight from the plane cache (0.2.4), which skips the trit decode + bit unpack that made the cold load slow."""
        self.lib = lib
        if planes is not None:
            self.rows, self.cols, pa, pb, lo, hi = planes
            self.pa = np.ascontiguousarray(pa, dtype=np.uint8); self.pb = np.ascontiguousarray(pb, dtype=np.uint8)
            self.lo = np.ascontiguousarray(lo, dtype=np.float16); self.hi = np.ascontiguousarray(hi, dtype=np.float16)
        else:
            self.rows, self.cols = rec["sign"].shape
            self.pa, self.pb = pack_planes(rec)
            self.lo = np.ascontiguousarray(rec["lo"].astype(np.float16)); self.hi = np.ascontiguousarray(rec["hi"].astype(np.float16))
        self.h = lib.phonon2_cpu_create(self.rows, self.cols, self.pa.ctypes.data, self.pb.ctypes.data, self.lo.ctypes.data, self.hi.ctypes.data, 0)
        assert self.h, "create failed"
        self.onedot_max_rel_perturbation = None
        if os.environ.get("PHONON2_CPU_ONEDOT", "0") == "1":
            self.onedot_max_rel_perturbation = float(lib.phonon2_cpu_enable_onedot(self.h))

    def gemm(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32); M = x.shape[0]
        y = np.empty((M, self.rows), dtype=np.float32)
        rc = self.lib.phonon2_cpu_gemm(self.h, x.ctypes.data, M, y.ctypes.data, self.rows)
        assert rc == 0, rc
        return y

    def dequant(self) -> np.ndarray:
        w = np.empty((self.rows, self.cols), dtype=np.float32)
        for n in range(self.rows):
            self.lib.phonon2_cpu_dequant_row(self.h, n, w[n].ctypes.data)
        return w

    def __del__(self):
        try:
            self.lib.phonon2_cpu_destroy(self.h)
        except Exception:
            pass


def reference_int8(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The kernel's arithmetic in numpy: per-row symmetric int8 activations, exact integer dots, one fp32 scale per (m, n).
    w is the exact fp32 dequantised weight; the split into lo*<xq,p_a> + d*<xq,p_b> is algebraically the same integer sum."""
    amax = np.abs(x).max(axis=1, keepdims=True); s = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    xq = np.clip(np.rint(x / s), -127, 127).astype(np.float64)
    return (s.astype(np.float64) * (xq @ w.astype(np.float64).T)).astype(np.float32)


# ------------------------------------------------------------------ torch module
def torch_module_class():
    import torch, torch.nn as nn

    class PackedFiveValueCPU(nn.Module):
        def __init__(self, lib, rec, bias, conv1d: bool):
            super().__init__()
            self.pm = rec if isinstance(rec, PackedMatrix) else PackedMatrix(lib, rec); self.conv1d = conv1d
            self.bias = None if bias is None else nn.Parameter(bias.detach().float().clone(), requires_grad=False)
            self.out_features, self.in_features = self.pm.rows, self.pm.cols

        def forward(self, x):
            if self.conv1d:
                x = x.transpose(1, 2)
            shp = x.shape; x2 = x.reshape(-1, shp[-1]).detach().float().contiguous().numpy()
            y = torch.from_numpy(self.pm.gemm(x2)).reshape(*shp[:-1], self.out_features)
            if self.bias is not None:
                y = y + self.bias
            return y.transpose(1, 2) if self.conv1d else y
    return PackedFiveValueCPU


def swap_encoder(model, raw_records: dict, fv_names: list[str], lib, threads: int | None = None) -> int:
    import torch.nn as nn
    Packed = torch_module_class()
    if threads:
        lib.phonon2_cpu_set_threads(int(threads))
    n = 0
    for name in fv_names:
        parent = model; parts = name.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        old = getattr(parent, parts[-1])
        bias = old.bias if getattr(old, "bias", None) is not None else None
        setattr(parent, parts[-1], Packed(lib, raw_records[name], bias, isinstance(old, nn.Conv1d))); n += 1
    return n


# ------------------------------------------------------------------ C greedy TDT loop (phonon2_cpu_tdt.c): int6 decoder tables exact, int8 activations
def default_tdt_library() -> Path:
    """The C TDT-loop binary for this platform, resolved by platform name (0007b.1: it used to be derived from the GEMM
    library's file name by string replacement, which was a no-op once the GEMM library became libphonon2_enc-*)."""
    override = os.environ.get("PHONON2_TDT_LIBRARY")
    if override:
        return Path(override)
    p = _plan()
    if p["refusal"]:
        raise RuntimeError(p["refusal"])
    base = p["base"] and p["detect"]["arch"] in _BASE_NEEDS_TDT
    name = _platform_name("phonon2_tdt", base=base)
    if base and not (name and (HERE / name).is_file()):
        from fermion._speech import _cpu_features
        raise RuntimeError(_cpu_features.missing_line(p["detect"], f"baseline TDT binary ({name})"))
    return HERE / (name or "libphonon2_tdt.so")


def shipped_binaries() -> dict:
    """Which of the three binaries are shipped AND pinned for this platform (the loader's own contract)."""
    pins = _pins(); out = {}
    try:
        p = _plan(); base = p["base"] and not p["refusal"]
    except Exception:
        base = False
    for key, prefix in (("gemm", "phonon2_cpu"), ("tdt", "phonon2_tdt"), ("enc", "phonon2_enc")):
        b = base and (key == "enc" or (key == "tdt" and _plan()["detect"]["arch"] in _BASE_NEEDS_TDT))
        name = _platform_name(prefix, base=b); ok = bool(name and (HERE / name).is_file() and name in pins)
        if key == "gemm" and base:
            enc = _platform_name("phonon2_enc", base=True); ok = bool(enc and (HERE / enc).is_file() and enc in pins)   # -base encoder carries the GEMM
        out[key] = ok
    return out


class CTdtDecoder:
    """Greedy TDT decode of one utterance in C: emb -> 2-layer LSTM -> projector -> joint head, per-vector int8 activations,
    the container's int6 decoder tables (exact).  39/40 texts identical to the torch loop on the M5 gate; loop cost 1.9 ms
    per audio-second (torch loop ≈ 35)."""

    def __init__(self, model, raw: dict, tensors: dict, vocab_size: int, threads: int, lib_path=None):
        tp = Path(lib_path) if lib_path else default_tdt_library()
        if not os.environ.get("PHONON2_TDT_LIBRARY") and lib_path is None:
            import hashlib
            pin = _pins().get(tp.name)
            if not tp.is_file() or pin is None or hashlib.sha256(tp.read_bytes()).hexdigest() != pin:
                raise RuntimeError(f"{tp.name}: missing or does not match the shipped sha256 pin; refusing to load")
        self.lib = ctypes.CDLL(str(tp))
        L = self.lib
        L.phonon2_tdt_create.restype = ctypes.c_void_p
        L.phonon2_tdt_create.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p] + [ctypes.c_int] * 2 + [ctypes.c_void_p] * 20
        L.phonon2_tdt_decode.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int); L.phonon2_tdt_decode.restype = ctypes.c_int
        L.phonon2_tdt_set_threads.argtypes = (ctypes.c_int,); L.phonon2_tdt_destroy.argtypes = (ctypes.c_void_p,)
        # 0.2.7 (TDT ABI 2): the same loop also hands back the frame index and duration of every emitted token.
        self.timed = hasattr(L, "phonon2_tdt_decode_timed")
        if self.timed:
            L.phonon2_tdt_decode_timed.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
            L.phonon2_tdt_decode_timed.restype = ctypes.c_int
        # 0.2.9 (TDT ABI 3): hotwords -- phonon2_tdt_set_bias arms the loop with the automaton tables of
        # fermion/_speech/hotwords.py (bonus only, blank never biased); disarmed, the loop is the ABI 2 loop.
        self.can_bias = self.timed and hasattr(L, "phonon2_tdt_set_bias")
        if self.can_bias:
            L.phonon2_tdt_set_bias.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_float)
            L.phonon2_tdt_set_bias.restype = ctypes.c_int
        cfg = model.config
        def Q(n): return np.ascontiguousarray(raw[n]["q"]), np.ascontiguousarray(raw[n]["scale"].astype(np.float16))
        def F(n): return np.ascontiguousarray(np.asarray(tensors[n]).astype(np.float16))
        emb_q, emb_s = Q("decoder.embedding.weight"); ih0, ih0s = Q("decoder.lstm.weight_ih_l0"); hh0, hh0s = Q("decoder.lstm.weight_hh_l0")
        ih1, ih1s = Q("decoder.lstm.weight_ih_l1"); hh1, hh1s = Q("decoder.lstm.weight_hh_l1"); pq, ps = Q("decoder.decoder_projector.weight"); hq, hs = Q("joint.head.weight")
        bih0, bhh0, bih1, bhh1, pb, hb = [F(n) for n in ("decoder.lstm.bias_ih_l0", "decoder.lstm.bias_hh_l0", "decoder.lstm.bias_ih_l1", "decoder.lstm.bias_hh_l1", "decoder.decoder_projector.bias", "joint.head.bias")]
        self._keep = [emb_q, emb_s, ih0, ih0s, bih0, hh0, hh0s, bhh0, ih1, ih1s, bih1, hh1, hh1s, bhh1, pq, ps, pb, hq, hs, hb]
        durs = (ctypes.c_int * len(cfg.durations))(*cfg.durations); self._durs = durs
        E = emb_q.shape[1]; H = hh0.shape[1]
        self.h = L.phonon2_tdt_create(emb_q.shape[0], E, H, hq.shape[0], len(cfg.durations), ctypes.addressof(durs), cfg.blank_token_id,
                                      getattr(cfg, "max_symbols_per_step", 10) or 10, *[a.ctypes.data for a in self._keep])
        assert self.h, "phonon2_tdt_create failed"
        L.phonon2_tdt_set_threads(int(threads)); self.H = H

    def decode(self, encp) -> list[int]:
        """encp: [T, H] float32 numpy (encoder_projector output for the true frames)."""
        e = np.ascontiguousarray(encp, dtype=np.float32); T = e.shape[0]
        out = np.zeros(max(64, 12 * T), np.int32)
        n = self.lib.phonon2_tdt_decode(self.h, e.ctypes.data, T, out.ctypes.data, out.shape[0])
        return [int(i) for i in out[:n]]

    def decode_timed(self, encp, bias=None):
        """(token ids, emission frame per token, predicted duration in frames per token); the two timing lists are None
        when the loaded binary has no timed entry point (an older override library), and the ids are then `decode()`'s.
        `bias`: a hotwords.HotwordAutomaton for this decode (needs `can_bias`); None = the unbiased loop."""
        if bias is not None and not self.can_bias:
            raise RuntimeError("the loaded TDT-loop binary has no hotword entry point (TDT ABI < 3)")
        if not self.timed:
            return self.decode(encp), None, None
        e = np.ascontiguousarray(encp, dtype=np.float32); T = e.shape[0]
        cap = max(64, 12 * T)
        out = np.zeros(cap, np.int32); fr = np.zeros(cap, np.int32); du = np.zeros(cap, np.int32)
        if bias is None:
            n = self.lib.phonon2_tdt_decode_timed(self.h, e.ctypes.data, T, out.ctypes.data, fr.ctypes.data, du.ctypes.data, cap)
        else:
            tb = [np.ascontiguousarray(a, dtype=np.int32) for a in (bias.table, bias.toks, bias.col, bias.depth)]
            S, K = tb[0].shape
            rc = self.lib.phonon2_tdt_set_bias(self.h, int(S), int(K), tb[0].ctypes.data, tb[1].ctypes.data, tb[2].ctypes.data,
                                               int(tb[2].shape[0]), tb[3].ctypes.data, float(bias.lam))
            if rc != 0:
                raise RuntimeError(f"phonon2_tdt_set_bias rejected the hotword tables (rc {rc})")
            try:
                n = self.lib.phonon2_tdt_decode_timed(self.h, e.ctypes.data, T, out.ctypes.data, fr.ctypes.data, du.ctypes.data, cap)
            finally:                                   # the tables are not copied by the library: disarm before they go
                self.lib.phonon2_tdt_set_bias(self.h, 0, 0, None, None, None, 0, None, 0.0)
        return [int(i) for i in out[:n]], [int(i) for i in fr[:n]], [int(i) for i in du[:n]]

    def __del__(self):
        try:
            self.lib.phonon2_tdt_destroy(self.h)
        except Exception:
            pass
