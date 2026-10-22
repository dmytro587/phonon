"""`fermion inspect` — container anatomy for the file the user has on disk.

WHY THIS COMMAND EXISTS
    In a float checkpoint "which connections are silenced" is not a readable
    property of the file; you have to guess it from magnitudes. In a Neutrino
    container the weight state IS the format, so occupancy is a fact you can
    read off the artifact in seconds. This command reports that fact, plus the
    byte budget and geometry of the file.

WHAT THIS COMMAND DELIBERATELY DOES NOT PRINT
    The walk below has to know where things are in the file; the OUTPUT never
    says. No record ordering, no plane/record structure, no header field
    layout, no byte offsets, no metadata encoding, no coder mechanism, no
    multi-level scale structure, no bits-per-weight figure, no occupancy split
    finer than the three states. Everything emitted is an aggregate statistic
    or a geometry number that is already on the model card. If you extend this
    file, keep that line: aggregates out, never a layout map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Projection names in the order the architecture defines them. This is model
# architecture (public: it is just a Qwen3-shaped block), not container layout.
PROJS = ("q", "k", "v", "o", "gate", "up", "down")
ATTN_PROJS = ("q", "k", "v", "o")
_CHUNK = 1 << 24


# ------------------------------------------------------------------ internals
def _pop_table():
    import numpy as np
    return np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def _bits(arr, table) -> int:
    """Set-bit count of a uint8 array (histogram . popcount, no big copies)."""
    import numpy as np
    h = np.bincount(arr, minlength=256)
    return int((h * table).sum())


def _reader(path: Path):
    """The loader's own container reader — inspect deliberately does not carry
    a second copy of the parsing logic."""
    try:
        from hf_ternary.loader import HEADER, Reader, read_header
    except ImportError:
        from fermion._vendor.hf_ternary.loader import (HEADER, Reader,
                                                       read_header)
    return Reader(path), read_header(path), HEADER.size


def _walk(path: Path, want=None, progress=False) -> dict:
    """Read-only walk of a v4 container, on top of the loader's reader.

    `want(layer, proj) -> bool` selects which projections get counted; the
    rest are stepped over without touching their bytes (that is what makes
    `--layer N` instant). Returns aggregates only — the caller never sees an
    offset or a record shape.
    """
    import numpy as np

    table = _pop_table()
    size = path.stat().st_size
    r, h, header_bytes = _reader(path)
    mm = r.mm
    arch = h.arch
    n_layers, hidden, n_heads, n_kv = h.n_layers, h.hidden, h.n_heads, h.n_kv_heads
    head_dim, rotary, inter = h.head_dim, h.rotary_dims, h.intermediate
    vocab, max_pos, eok = h.vocab, h.max_position, h.eok

    out = {
        "geometry": {
            "layers": n_layers, "hidden": hidden, "q_heads": n_heads,
            "kv_heads": n_kv, "head_dim": head_dim, "rotary_dims": rotary,
            "intermediate": inter, "vocab": vocab, "max_context": max_pos,
            "tied_embeddings": (eok == 1),
        },
        "bytes": {"ternary_lane": 0, "embeddings_int8": 0,
                  "metadata": header_bytes, "norms": 0},
        "params": {"ternary_projection_weights": 0,
                   "embedding_params_int8": 0, "norm_params_fp32": 0},
        # occ[layer][proj] = dict(minus/zero/plus/weights); measured only
        "occ": [{} for _ in range(n_layers)],
        "layer_lane_bytes": [0] * n_layers,
        "attn_lane_bytes": [0] * n_layers,
        "mlp_lane_bytes": [0] * n_layers,
        "file_bytes": size,
    }
    r.off = header_bytes

    def int8_record():
        rows, cols, _cols_pad, stride = r.unpack("<4i")
        out["bytes"]["metadata"] += 16
        out["bytes"]["embeddings_int8"] += rows * stride
        out["params"]["embedding_params_int8"] += rows * cols
        r.skip(rows * stride)
        r.skip(rows * 8)
        out["bytes"]["metadata"] += rows * 8

    def count_states(base: int, rows: int, cols: int, stride: int):
        """Set bits in the two sign lanes, over true (unpadded) columns."""
        vb = cols // 8
        masked = stride != vb
        plane = rows * stride
        counts = [0, 0]
        rows_per = max(1, _CHUNK // max(1, stride))
        for lane in (0, 1):
            for r0 in range(0, rows, rows_per):
                r1 = min(rows, r0 + rows_per)
                blk = np.frombuffer(
                    mm, dtype=np.uint8, count=(r1 - r0) * stride,
                    offset=base + lane * plane + r0 * stride)
                if masked:
                    blk = blk.reshape(r1 - r0, stride)[:, :vb].ravel()
                counts[lane] += _bits(blk, table)
        return counts

    def trit_record(measure: bool):
        rows, cols, _cols_pad, stride = r.unpack("<4i")
        out["bytes"]["metadata"] += 16
        plane = rows * stride
        stat = None
        if measure:
            n_pos, n_neg = count_states(r.off, rows, cols, stride)
            nw = rows * cols
            stat = {"minus": n_neg / nw, "zero": (nw - n_pos - n_neg) / nw,
                    "plus": n_pos / nw, "weights": nw}
        r.skip(3 * plane)
        r.skip(16 * rows)                    # per-row transport metadata
        out["bytes"]["metadata"] += 16 * rows
        (has_bias,) = r.unpack("<i")
        out["bytes"]["metadata"] += 4
        if has_bias:
            r.skip(4 * rows)
            out["bytes"]["metadata"] += 4 * rows
        out["bytes"]["ternary_lane"] += 3 * plane
        out["params"]["ternary_projection_weights"] += rows * cols
        return stat, 3 * plane

    if eok == 2:
        int8_record()
    # per-layer norm vectors, sized as the loader reads them for this arch
    norm_floats = {1: 2 * hidden,
                   2: 2 * hidden + hidden + inter,
                   3: 2 * hidden + 2 * head_dim}[arch]
    norm_per_layer = norm_floats * 4
    for li in range(n_layers):
        out["bytes"]["norms"] += norm_per_layer
        out["params"]["norm_params_fp32"] += norm_floats
        r.skip(norm_per_layer)
        for pk in PROJS:
            measure = True if want is None else bool(want(li, pk))
            stat, lane_bytes = trit_record(measure)
            if stat is not None:
                out["occ"][li][pk] = stat
            out["layer_lane_bytes"][li] += lane_bytes
            if pk in ATTN_PROJS:
                out["attn_lane_bytes"][li] += lane_bytes
            else:
                out["mlp_lane_bytes"][li] += lane_bytes
        if progress:
            print(f"  reading layer {li + 1}/{n_layers}",
                  end="\r", file=sys.stderr, flush=True)
    out["bytes"]["norms"] += hidden * 4
    out["params"]["norm_params_fp32"] += hidden
    r.skip(hidden * 4)
    int8_record()
    consumed = r.off
    r.close()
    if consumed != size:
        raise SystemExit(f"{path}: walk consumed {consumed} of {size} bytes — "
                         "container looks truncated or is a newer revision")
    if progress:
        print(" " * 40, end="\r", file=sys.stderr)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------- public summary
def anatomy(path: Path, layer: int | None = None, projection: str | None = None,
            sha: bool = False, progress: bool = False) -> dict:
    """Publishable aggregates for one container. No layout information."""
    t0 = time.time()
    if layer is None and projection is None:
        want = None
    else:
        def want(li, pk):
            return ((layer is None or li == layer)
                    and (projection is None or pk == projection))
    raw = _walk(path, want=want, progress=progress)
    g = raw["geometry"]
    n_layers = g["layers"]
    if layer is not None and not (0 <= layer < n_layers):
        raise SystemExit(f"--layer {layer} out of range (model has "
                         f"{n_layers} layers, 0-{n_layers - 1})")
    b = raw["bytes"]
    p = raw["params"]
    total_params = sum(p.values())
    lane_total = sum(b.values())
    if lane_total != raw["file_bytes"]:
        raise SystemExit("byte lanes do not sum to file size")

    def agg(rows):
        w = sum(r["weights"] for r in rows)
        if not w:
            return None
        return {st: sum(r[st] * r["weights"] for r in rows) / w
                for st in ("minus", "zero", "plus")} | {"weights": w}

    measured = [r for L in raw["occ"] for r in L.values()]
    per_layer = [agg(list(L.values())) for L in raw["occ"]]
    per_proj = {pk: agg([L[pk] for L in raw["occ"] if pk in L])
                for pk in PROJS}
    per_proj = {k: v for k, v in per_proj.items() if v}

    kv_tok_fp16 = n_layers * g["kv_heads"] * g["head_dim"] * 2 * 2
    # int8 (0.1.10) is not simply half of fp16: the runtime stores one fp32
    # scale per cached head-vector, so it is (hd + 4) bytes per vector, not
    # hd. Mirrors the native runtime's own arithmetic:
    #   2 * L * n_kv * (hd * elt + (4 if int8 else 0))
    kv_tok_int8 = n_layers * g["kv_heads"] * (g["head_dim"] + 4) * 2
    ctxs = [4096, 32768, g["max_context"], 131072]
    ctxs = sorted(set(c for c in ctxs if c > 0))

    doc = {
        "file": {
            "name": path.name,
            "path": str(path),
            "bytes": raw["file_bytes"],
            "walk_seconds": round(time.time() - t0, 2),
        },
        "geometry": g | {
            "attention_grouping": f"{g['q_heads']} query heads share "
                                  f"{g['kv_heads']} kv heads "
                                  f"(GQA {g['q_heads'] // g['kv_heads']}:1)",
        },
        "parameters_by_tensor_class": p | {"total": total_params},
        "weight_state_occupancy": {
            "description": "Share of quantized projection weights in each of "
                           "the three weight states, over true (unpadded) "
                           "weights.",
            "global": agg(measured),
            "by_layer": per_layer,
            "by_projection": per_proj,
            "scope": ("all layers, all projections" if want is None else
                      f"layer={layer if layer is not None else 'all'}, "
                      f"projection={projection or 'all'}"),
        },
        "byte_budget": {
            "lanes": {
                "ternary_weight_lane": b["ternary_lane"],
                "token_embeddings_int8": b["embeddings_int8"],
                "metadata": b["metadata"],
                "norms_fp32": b["norms"],
            },
            "lane_shares": {k: v / raw["file_bytes"] for k, v in (
                ("ternary_weight_lane", b["ternary_lane"]),
                ("token_embeddings_int8", b["embeddings_int8"]),
                ("metadata", b["metadata"]),
                ("norms_fp32", b["norms"]))},
            "weight_lane_bytes_per_layer": raw["layer_lane_bytes"],
            "attention_share_of_layer": [
                a / (a + m) for a, m in zip(raw["attn_lane_bytes"],
                                            raw["mlp_lane_bytes"])],
            "mlp_share_of_layer": [
                m / (a + m) for a, m in zip(raw["attn_lane_bytes"],
                                            raw["mlp_lane_bytes"])],
        },
        "kv_cache": {
            "bytes_per_token_fp16": kv_tok_fp16,
            "bytes_per_token_fp32": kv_tok_fp16 * 2,
            "bytes_per_token_int8": kv_tok_int8,
            "totals_fp16": {str(c): kv_tok_fp16 * c for c in ctxs},
            "totals_fp32": {str(c): kv_tok_fp16 * 2 * c for c in ctxs},
            "totals_int8": {str(c): kv_tok_int8 * c for c in ctxs},
            "note": "fp16 is the default since 0.1.10; fp32 is the "
                    "reproducibility anchor and int8 the long-context option "
                    "(--kv-dtype). int8 carries one fp32 scale per cached "
                    "head-vector, so it is slightly above a quarter of fp32.",
        },
    }
    if sha:
        doc["file"]["sha256"] = sha256_file(path)
    return doc


# ------------------------------------------------------------------ rendering
def _n(x) -> str:
    return f"{x:,}"


def _gib(x) -> str:
    if x >= 2 ** 30:
        return f"{x / 2 ** 30:.2f} GiB"
    if x >= 2 ** 20:
        return f"{x / 2 ** 20:.1f} MiB"
    return f"{x / 2 ** 10:.1f} KiB"


def _bar(frac: float, width: int = 28) -> str:
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def _pct(x) -> str:
    return f"{100 * x:6.2f}%"


def render(doc: dict, layer: int | None, projection: str | None) -> str:
    L = []
    f, g = doc["file"], doc["geometry"]
    p = doc["parameters_by_tensor_class"]
    occ = doc["weight_state_occupancy"]
    bb = doc["byte_budget"]
    kv = doc["kv_cache"]
    add = L.append

    add(f"fermion inspect · {f['name']}")
    line = (f"  {_n(f['bytes'])} bytes ({_gib(f['bytes'])})"
            f" · walked in {f['walk_seconds']:.1f}s")
    if "sha256" in f:
        line += f" · sha256 {f['sha256'][:16]}…"
    add(line)
    add("  every byte in the file is accounted for by the lanes below")
    add("")

    add("GEOMETRY")
    add(f"  layers {g['layers']} · hidden {_n(g['hidden'])} · "
        f"intermediate {_n(g['intermediate'])}")
    add(f"  {g['attention_grouping']} · head_dim {g['head_dim']} · "
        f"rotary {g['rotary_dims']}")
    add(f"  vocab {_n(g['vocab'])} · max context {_n(g['max_context'])} · "
        f"embeddings {'tied' if g['tied_embeddings'] else 'untied'}")
    add("")

    add("PARAMETERS BY TENSOR CLASS")
    tot = p["total"]
    for label, key in (("ternary projections", "ternary_projection_weights"),
                       ("token embeddings (int8)", "embedding_params_int8"),
                       ("norms (fp32)", "norm_params_fp32")):
        add(f"  {label:<24}{_n(p[key]):>16}  {100 * p[key] / tot:6.2f}%")
    add(f"  {'total':<24}{_n(tot):>16}")
    add("")

    if occ["global"]:
        gl = occ["global"]
        add(f"WEIGHT-STATE OCCUPANCY  ({_n(gl['weights'])} weights measured)")
        if occ["scope"] != "all layers, all projections":
            add(f"  scope: {occ['scope']}")
        for st in ("minus", "zero", "plus"):
            add(f"  {st:<6}{_pct(gl[st])}  {_bar(gl[st])}")
        add(f"  {100 * gl['zero']:.2f}% of this model's connections are "
            "silenced, and that is")
        add("  a property you can read off the file — not an estimate of one.")
        add("")

    byproj = occ["by_projection"]
    if byproj and (layer is not None or len(byproj) > 1):
        title = ("OCCUPANCY BY PROJECTION" if layer is None else
                 f"OCCUPANCY BY PROJECTION · layer {layer}")
        add(title + ("  (all layers)" if layer is None else ""))
        add(f"  {'proj':<6}{'minus':>8}{'zero':>9}{'plus':>8}")
        for pk, v in byproj.items():
            add(f"  {pk:<6}{_pct(v['minus'])[1:]:>8}{_pct(v['zero'])[1:]:>9}"
                f"{_pct(v['plus'])[1:]:>8}")
        add("")

    rows = [(i, v) for i, v in enumerate(occ["by_layer"]) if v]
    if len(rows) > 1:
        what = f"{projection} projection" if projection else "all projections"
        add(f"ZERO-STATE OCCUPANCY BY LAYER  ({what})")
        cells = [f"L{i:02d} {100 * v['zero']:.1f}%" for i, v in rows]
        for i in range(0, len(cells), 6):
            add("  " + "  ".join(cells[i:i + 6]))
        lo = min(rows, key=lambda r: r[1]["zero"])
        hi = max(rows, key=lambda r: r[1]["zero"])
        add(f"  low {100 * lo[1]['zero']:.2f}% (layer {lo[0]}) · "
            f"high {100 * hi[1]['zero']:.2f}% (layer {hi[0]}) · "
            f"spread {100 * (hi[1]['zero'] - lo[1]['zero']):.2f} pp")
        add("")

    add("BYTE BUDGET")
    lanes = bb["lanes"]
    for label, key in (("ternary weight lane", "ternary_weight_lane"),
                       ("token embeddings (int8)", "token_embeddings_int8"),
                       ("metadata", "metadata"),
                       ("norms (fp32)", "norms_fp32")):
        add(f"  {label:<24}{_n(lanes[key]):>16}  "
            f"{100 * bb['lane_shares'][key]:6.2f}%  ({_gib(lanes[key])})")
    per = bb["weight_lane_bytes_per_layer"]
    uniform = len(set(per)) == 1
    a0 = bb["attention_share_of_layer"][0]
    m0 = bb["mlp_share_of_layer"][0]
    if uniform:
        add(f"  weight lane per layer   {_n(per[0]):>16}  "
            f"(identical for all {len(per)} layers)")
    else:
        add(f"  weight lane per layer   {_n(min(per)):>16} … {_n(max(per))}")
    add(f"  of which attention {100 * a0:.1f}% · MLP {100 * m0:.1f}%")
    add("")

    add("KV CACHE")
    add(f"  {_n(kv['bytes_per_token_fp16'])} bytes per token (fp16, default) · "
        f"{_n(kv['bytes_per_token_fp32'])} (fp32) · "
        f"{_n(kv['bytes_per_token_int8'])} (int8)")
    for c, v in kv["totals_fp16"].items():
        tag = " (header max)" if int(c) == g["max_context"] else ""
        add(f"  {_n(int(c)):>9} ctx   {_gib(v):>10} fp16   "
            f"{_gib(kv['totals_fp32'][c]):>10} fp32   "
            f"{_gib(kv['totals_int8'][c]):>10} int8{tag}")
    add("  select with --kv-dtype {fp16,fp32,int8}")
    return "\n".join(L)


# ------------------------------------------------------------------ cli glue
def add_parser(sub, common) -> None:
    ap = sub.add_parser(
        "inspect", parents=[common],
        help="print the anatomy of a container: weight-state occupancy, "
             "byte budget, geometry, KV cost")
    ap.add_argument("--layer", type=int, default=None,
                    help="restrict occupancy to one layer (fast: skips the "
                         "rest of the file)")
    ap.add_argument("--projection", default=None, choices=list(PROJS),
                    help="restrict occupancy to one projection")
    ap.add_argument("--sha256", action="store_true",
                    help="also hash the file (adds a full read)")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="machine-readable output")


def run(args, kind: str, path: Path) -> int:
    if kind == "tiny":
        raise SystemExit("inspect works on TRTC v4 containers, not tiny "
                         "QRB8/QRL8 artifacts")
    if kind == "trtc-dir":
        from fermion._container import find_container
        path = find_container(path)
    doc = anatomy(path, layer=args.layer, projection=args.projection,
                  sha=args.sha256, progress=not args.as_json)
    if args.as_json:
        print(json.dumps(doc, indent=1))
    else:
        print(render(doc, args.layer, args.projection))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual use
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--projection")
    ap.add_argument("--sha256", action="store_true")
    ap.add_argument("--json", dest="as_json", action="store_true")
    a = ap.parse_args()
    sys.exit(run(a, "trtc-bin", Path(a.path)))
