"""fermion: five-value sub-2-bit LLM loader + CLI.

Re-exports the hf_ternary integration (TRTC v4 -> native Transformers
models). If a standalone hf_ternary install is present it is used; otherwise
the vendored copy (verbatim, sha-recorded in _vendor/) is loaded.

LAZY SINCE 0.1.25. `import fermion` imports nothing heavy: no torch, no
transformers, no MLX. The language-model loader is imported the first time
one of its symbols is used (`fermion.load_trtc_v4`, `fermion.read_header`,
`from fermion import TrtcV4ForCausalLM`, ...) or when
`fermion.language_model_loader()` is called explicitly; that first use also
registers the `trtc_v4` model type with transformers' Auto classes, exactly as
`import fermion` did before. Code that relied on the registration as a side
effect of the bare import — a transformers `from_pretrained` on a trtc_v4
directory with no fermion symbol touched first — should call
`fermion.language_model_loader()` once. The speech engines import on first
use as well. `fermion --version` went from ~2.5 s to well under 0.3 s.
"""
from __future__ import annotations

from fermion._brand import BRAND, DEFAULT_MODEL_ID, VERSION

_LLM_EXPORTS = (
    "TritLinearInference", "PackedInt8Embedding", "PackedInt8Linear",
    "TrtcV4Config", "TrtcV4ForCausalLM", "load_trtc_v4", "read_header",
    "register", "write_transformers_config", "resident_buffer_bytes")
_LLM_IMPORT_ERROR: BaseException | None = None
_LLM = None                     # the loader module, once imported


def language_model_loader():
    """Import the language-model loader now and return its module.

    Registers `trtc_v4` with transformers' Auto classes (once). Raises
    ImportError, with the original failure as its cause, when torch or
    transformers cannot be imported — the speech commands are unaffected by
    that and never call this.
    """
    global _LLM, _LLM_IMPORT_ERROR
    if _LLM is not None:
        return _LLM
    if _LLM_IMPORT_ERROR is not None:
        raise ImportError(_llm_error_text()) from _LLM_IMPORT_ERROR
    from fermion import _compat
    # Before torch/transformers load: a torchaudio built for another torch
    # must not take the import down with it (fermion never uses torchaudio).
    _compat.prepare_torch_stack()
    try:
        try:
            import hf_ternary as module
            for _name in _LLM_EXPORTS:          # a foreign hf_ternary that lacks
                getattr(module, _name)          # a symbol -> vendored copy
        except (ImportError, AttributeError):
            from fermion._vendor import hf_ternary as module
    except Exception as exc:                                   # noqa: BLE001
        # The loader needs torch + transformers. When that stack is broken (a
        # shared-library error out of some third package, an incompatible
        # transformers), `import fermion`, `fermion --version` and the speech
        # verbs keep working; the failure surfaces here, with its cause.
        _LLM_IMPORT_ERROR = exc
        raise ImportError(_llm_error_text()) from exc
    _LLM = module
    for _name in _LLM_EXPORTS:
        globals()[_name] = getattr(module, _name)
    return module


def _llm_error_text() -> str:
    return (f"fermion's language-model loader could not be imported "
            f"({type(_LLM_IMPORT_ERROR).__name__}: {_LLM_IMPORT_ERROR}); "
            f"the speech commands are unaffected")


def load_speech(model: str, *, quiet: bool = True):
    """Load a speech model on this machine's engine and return it (0.2.9).

    `model` is what `fermion transcribe` takes: `phonon-2`, `phonon-1`, a repo id or a local model directory. The
    engine is chosen exactly as for the CLI (MLX on Apple silicon, the CPU engine elsewhere; `FERMION_DEVICE`).

        speech = fermion.load_speech("phonon-2")
        text, decode_s, audio_s = speech.transcribe("call.wav", hotwords=["Kushal", "Sigil"])
        speech.set_hotwords(["Kushal", "Sigil"], lam=2.0)    # or for every later call

    `transcribe`, `transcribe_detailed` (text + segments + word timestamps) and `transcribe_array` (16 kHz mono float32)
    take `hotwords=[...]` (at most 25 names and terms on Phonon-2) and `hotword_lambda=` (the bonus per matching word
    piece, default 2.0) for that call; an empty list is the plain decode.
    """
    from fermion._speech import backends, fetch
    from fermion.transcribe import _resolve
    repo, key, pin, local_dir = _resolve(model)
    kind = backends.resolve("`fermion.load_speech`")
    backends.require_engine_for(kind, pin.get("backend", ""))
    model_dir = local_dir if local_dir is not None else fetch.ensure(repo, key, pin)
    return backends.load(kind, model_dir, profile=key, backend=pin["backend"], quiet=quiet)


def __getattr__(name):                                          # PEP 562
    if name in _LLM_EXPORTS:
        return getattr(language_model_loader(), name)
    raise AttributeError(f"module 'fermion' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LLM_EXPORTS))


__version__ = VERSION
__all__ = [
    "TritLinearInference", "PackedInt8Embedding", "PackedInt8Linear",
    "TrtcV4Config", "TrtcV4ForCausalLM", "load_trtc_v4", "read_header",
    "register", "write_transformers_config", "resident_buffer_bytes",
    "language_model_loader", "load_speech", "BRAND", "DEFAULT_MODEL_ID", "VERSION",
]
