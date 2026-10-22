"""Vendored Phonon-2 MLX runtime (five-value Parakeet-TDT over mlx-audio 0.4.6).

Four modules, byte-identical to `local_stt/phonon2/` in the engine tree
(shas in `SHA256SUMS.txt`, asserted by the selftest):

    fermion_container.py     reads the `fermion-five-value-parakeet-v1` container
    hf_to_mlx_parakeet.py    HF ParakeetForTDT names -> mlx-audio names (+ conv/LSTM layout)
    packed_runtime.py        PackedFiveValueLinear: two native 2-bit affine QMMs per linear
    load_parakeet.py         load_parakeet(container, config): the packed mlx-audio ParakeetTDT
    fast_decode.py           post-load speed levers; the default `tdt16` keeps the greedy TDT loop on the
                             GPU (one host sync per 16 steps) with tokens and timings identical to
                             mlx-audio's loop (400/400 utterances + 20 long holds, runtime lane 09-26/28)

They import each other by bare name, exactly like `_engine/`; the same flat-name
importer discipline applies, so they are loaded from this directory.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ("fermion_container", "hf_to_mlx_parakeet", "packed_runtime", "load_parakeet", "fast_decode")


def _install(name: str):
    mod = sys.modules.get(name)
    if mod is not None:
        if Path(getattr(mod, "__file__", "") or "").resolve().parent != HERE:
            raise ImportError(f"flat module name {name!r} is already bound to "
                              f"{getattr(mod, '__file__', '?')}, not our vendored copy")
        return mod
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def load(name: str = "load_parakeet"):
    if name not in MODULES:
        raise ImportError(f"{name!r} is not a vendored Phonon-2 engine module")
    for dep in MODULES:
        if dep == name:
            break
        _install(dep)
    return _install(name)


def vendored_files() -> list[Path]:
    return [HERE / f"{n}.py" for n in MODULES]
