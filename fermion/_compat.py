"""Environment shields that run right before the heavy stack loads.

Since 0.1.25 `import fermion` loads no torch, transformers or MLX: the
language-model loader and both speech engines import on first use, so
`fermion --version`, `--help` and `fermion models` cost a few tens of
milliseconds. The shields therefore no longer run at import time; they run
from `prepare_torch_stack()`, which every path that is about to import torch
or transformers calls first — the lazy language-model loader in
`fermion/__init__.py`, and the engine loaders in `fermion/_speech/backends.py`,
`fermion/_speech/engine.py` and `fermion/_speech/_engine_cpu/engine.py`.
The shield's behaviour is unchanged; only the moment it runs moved.
"""
from __future__ import annotations

import importlib.util
import sys

#: One human sentence when the torchaudio shield fired, for the CLI to print
#: once on stderr; None when torchaudio is absent or loads fine.
TORCHAUDIO_NOTE: str | None = None

_CHECKED = False
_PREPARED = False
#: Set by the CLI (`cli.main()`): print a shield's note on stderr the moment
#: it fires. Library users (`import fermion`) read `TORCHAUDIO_NOTE` instead.
_ANNOUNCE = False
_ANNOUNCED = False


def _dist_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:                                       # noqa: BLE001
        return "?"


def _print_note() -> None:
    global _ANNOUNCED
    if TORCHAUDIO_NOTE and _ANNOUNCE and not _ANNOUNCED:
        _ANNOUNCED = True
        from fermion._brand import BRAND
        print(f"[{BRAND}] {TORCHAUDIO_NOTE}", file=sys.stderr, flush=True)


def announce_notes() -> None:
    """CLI mode: shield notes go to stderr as they happen (once each)."""
    global _ANNOUNCE
    _ANNOUNCE = True
    _print_note()


def prepare_torch_stack() -> None:
    """Run every shield once, immediately before torch/transformers first load.

    Idempotent and cheap after the first call. Every code path that imports
    torch or transformers on the user's behalf calls this first, so a broken
    third-party package is masked before it can take the import down — the
    same protection `import fermion` gave in 0.1.24, now paid only by the
    commands that actually load a model.
    """
    global _PREPARED
    if _PREPARED:
        return
    _PREPARED = True
    shield_broken_torchaudio()


def _reset_transformers_probes() -> None:
    """Make an already-imported transformers forget that torchaudio looked
    installed.

    Since 0.1.25 the shield runs on first heavy use, not at `import fermion`,
    so a program may have imported transformers first. transformers 5 wraps
    `is_torchaudio_available()` (and `is_speech_available()`, which calls it)
    in `functools.lru_cache`, and `import transformers` itself primes that
    cache with True while torchaudio is still findable. Left alone, the next
    `import transformers.audio_utils` (the vendored speech processor, the
    language-model loader) would trust the cached True and run
    `import torchaudio` into our `None` sentinel — a ModuleNotFoundError where
    0.1.24 worked (found 2026-09-07 on the DLVM image). Clearing every cached
    probe on `transformers.utils.import_utils` is safe: they are pure
    environment lookups and recompute to the same values, except the
    torchaudio ones, which now read the sentinel and return False.
    transformers < 5 kept a plain module flag instead; reset that too.
    """
    iu = sys.modules.get("transformers.utils.import_utils")
    if iu is None:
        return
    for name in dir(iu):
        fn = getattr(iu, name, None)
        clear = getattr(fn, "cache_clear", None)
        if callable(fn) and callable(clear):
            try:
                clear()
            except Exception:                       # noqa: BLE001
                pass
    if hasattr(iu, "_torchaudio_available"):
        try:
            iu._torchaudio_available = False
        except Exception:                           # noqa: BLE001
            pass


def shield_broken_torchaudio() -> str | None:
    """Hide a torchaudio that cannot load against the installed torch.

    fermion never imports torchaudio. transformers' `audio_utils` does, at
    module load, whenever torchaudio merely LOOKS installed
    (`is_torchaudio_available()` is a find_spec probe) — and both the
    language-model loader (via `BitNetConfig` -> `modeling_layers` ->
    `processing_utils` -> `audio_utils`) and the speech engines (the vendored
    Qwen3-ASR processor) pull `audio_utils` in. A torchaudio built for a
    different torch — GCP Deep Learning VM images ship torch 2.9 next to
    torchaudio 2.11 — then raises `OSError: Could not load this library:
    .../_torchaudio.abi3.so` out of our import (reproduced 2026-09-06,
    reports/pip_linux_import_repro_20260906.md).

    Marking the module absent in `sys.modules` (`None`) is the documented
    import-system idiom: `importlib.util.find_spec("torchaudio")` returns None,
    so transformers skips the import, and any stray `import torchaudio` raises
    a plain ImportError instead of a shared-library error. Runs once.
    """
    global TORCHAUDIO_NOTE, _CHECKED
    if _CHECKED:
        return TORCHAUDIO_NOTE
    _CHECKED = True
    if "torchaudio" in sys.modules:                 # already imported (healthy)
        return None
    try:
        if importlib.util.find_spec("torchaudio") is None:
            return None
    except (ImportError, ValueError):
        return None
    try:
        import torchaudio                           # noqa: F401
        return None
    except Exception as exc:                        # noqa: BLE001  OSError (ABI), ImportError, RuntimeError
        for name in [m for m in sys.modules
                     if m == "torchaudio" or m.startswith("torchaudio.")]:
            del sys.modules[name]
        sys.modules["torchaudio"] = None
        _reset_transformers_probes()
        TORCHAUDIO_NOTE = (
            f"torchaudio {_dist_version('torchaudio')} is installed but does "
            f"not load with torch {_dist_version('torch')} "
            f"({type(exc).__name__}); it is ignored. fermion does not use "
            f"torchaudio; `pip uninstall torchaudio` removes this note.")
        _print_note()
        return TORCHAUDIO_NOTE
