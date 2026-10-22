"""`fermion describe` — what this machine is and which Phonon-2 CPU kernel tier it gets (0.2.5).

Prints the host, the processor features the engine detected (dotprod / i8mm / SVE on Arm; SSE4.1, AVX2, AVX-512 BW and
VNNI, AMX on x86-64), how they were read, the tier chosen and the kernel binaries that tier loads.  Nothing is downloaded
and no model is loaded; the selected kernel library is opened once to report its own view (`--no-load` skips even that).

    fermion describe
    fermion describe --json
    phonon describe            (same output)

Registered EARLY (see `_tools.EARLY`): there is no model to resolve.
"""
from __future__ import annotations

import json
import platform

from fermion._brand import BRAND, VERSION


def add_parser(sub, common) -> None:
    p = sub.add_parser("describe", help="show this machine's CPU features and the speech kernel tier it runs")
    p.add_argument("--json", action="store_true", help="emit one JSON object")
    p.add_argument("--no-load", action="store_true", help="do not open the selected kernel library for its own report")


def _yes(v) -> str:
    return "yes" if v else "no"


def render(d: dict) -> str:
    h = d["host"]; f = d["features"]
    lines = ["", f"  {BRAND} {VERSION} describe",
             f"  {'host':<12}{h['system']} {h['machine']} · {h['cpu'] or 'cpu model unknown'} · {h['cores']} logical cores · Python {h['python']}"]
    if platform.machine().lower() in ("aarch64", "arm64"):
        feats = ", ".join(f"{k} {_yes(f.get(k))}" for k in ("neon", "dotprod", "i8mm", "sve", "lse"))
    else:
        feats = ", ".join(f"{k} {_yes(f.get(k))}" for k in ("sse4.1", "avx2", "avx512bw", "avx512-vnni", "amx-int8"))
    lines.append(f"  {'features':<12}{feats}  (read via {d['detection']})")
    lines.append(f"  {'tier':<12}{d['tier']}" + ("  (baseline binaries)" if d.get("baseline_binaries") else ""))
    b = d.get("binaries") or {}
    if isinstance(b, dict) and "error" not in b:
        for key, info in b.items():
            lines.append(f"  {'  ' + key:<12}{info['file']}  shipped {_yes(info['shipped'])}, pinned {_yes(info['pinned'])}")
    elif b:
        lines.append(f"  {'binaries':<12}{b.get('error')}")
    sel = d.get("selected")
    if sel:
        lines.append(f"  {'selected':<12}{sel['gemm_encoder']} + {sel['tdt']}")
    lib = d.get("library")
    if lib:
        if "error" in lib:
            lines.append(f"  {'library':<12}did not load: {lib['error']}")
        else:
            extra = f", C sees features {lib['c_features']} (build {lib['c_build_features']}), Python sees {lib['python_features']}" if "c_features" in lib else ""
            lines.append(f"  {'library':<12}kernel {lib['kernel_id']} = {lib['kernel']}{extra}")
    if d.get("refusal"):
        lines.append(f"  {'note':<12}{d['refusal']}")
    lines.append("")
    return "\n".join(lines)


def run_early(args) -> int:
    from fermion._speech import _cpu_features
    from fermion.speech import _console_safe
    d = _cpu_features.describe(load=not getattr(args, "no_load", False))
    if getattr(args, "json", False):
        print(json.dumps(d, indent=1, default=str))
    else:
        print(_console_safe(render(d)))                     # Windows cp1252 consoles cannot show the middle dot
    return 0


def run(args, kind, path) -> int:                           # pragma: no cover
    return run_early(args)
