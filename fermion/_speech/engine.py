"""Load a Phonon profile and turn audio into text.

This is the speech twin of `fermion/native.py`'s `NativeModel`: it owns the
model's lifecycle and exposes the two calls the rest of the package needs
(`load()` and `transcribe()`), so `transcribe`, `serve` and `models` need no
branching beyond one dispatch each.

WHY THIS FILE DOES NOT USE `stt.py`
------------------------------------
The staged public tree's `stt.py` is a self-contained CLI, and its model paths
are module constants derived from its own location (`ROOT = Path(__file__)
.resolve().parent`, then `PARITY_MODEL = ROOT / "model_v18_mlx_quint5"`).
Vendored into a wheel, `ROOT` becomes `site-packages/fermion/_speech/_engine/`
and every one of those constants points at a directory that does not and should
not exist. So `stt.py` is deliberately NOT vendored; this module calls
`optimized_v18.load_optimized_v18()` with an explicit path instead, which is
exactly what `stt.py:_load_model_inner()` does for the fold backends.

The decode call below is `stt.py:transcribe_array()`, verbatim in its
parameters — `language="English"`, `temperature=0.0`,
`repetition_penalty=1.05`, `repetition_context_size=96`,
`min_chunk_duration=0.5`, and the same `token_budget()`.

`repetition_penalty` is there for repetition protection. Real audio — a live
microphone, a long recording, a pause, a stretch of silence or noise — can put
an autoregressive decoder into a loop that emits the same phrase until the
token budget runs out, and a penalty over a bounded context window is what
stops that. It is the reason the CLI and the app decode identically: the
parameters that keep long and live audio well-behaved are not something to
vary between one entry point and another.

LONG AUDIO (0.1.23). That call is made per WINDOW, not per file: audio over
35 s is cut into 25-35 s windows at pauses by `_speech/segment.py` and each
window gets its own token budget. Through 0.1.22 the whole file went into one
call with one 384-token budget, which truncated anything over ~2 minutes and
degenerated past ~3. Audio up to 35 s takes the unchanged single call.
"""
from __future__ import annotations

import errno
import math
import os
import sys
import time
from pathlib import Path

from fermion._brand import BRAND

SAMPLE_RATE = 16_000


def token_budget(seconds: float) -> int:
    """`stt.py:token_budget()`, verbatim.

    English speech rarely needs >5 decoder tokens/s. The modest headroom is
    deliberate: it stops a bad silence/loop decode before it can run away.
    """
    return max(24, min(384, int(math.ceil(seconds * 7.0)) + 20))


def _long_audio_note() -> str:
    """One sentence for `describe()`/`/health` on how long files are decoded."""
    from fermion._speech import segment
    return (f"audio over {segment.SINGLE_SHOT_MAX_S:.0f} s is decoded in "
            f"{segment.BAND_MIN_S:.0f}-{segment.BAND_MAX_S:.0f} s windows cut "
            f"at pauses, each with its own token budget")


class SpeechModel:
    """A loaded Phonon profile. One instance holds the weights."""

    kind = "speech"

    def __init__(self, model, *, path: Path, profile: str, backend: str,
                 decode: dict, load_seconds: float):
        self.model = model
        self.path = Path(path)
        self.profile = profile
        self.backend = backend
        self.decode = dict(decode)
        self.load_seconds = load_seconds
        # Opt-in vocabulary bias (`--hotwords`).  Empty = the unchanged decode,
        # byte for byte.  Set by the CLI / API through `set_hotwords()`.
        self.hotwords: list[str] = []
        self.hotword_strength: float = 4.0

    def set_hotwords(self, words, strength: float | None = None, *,
                     lam: float | None = None) -> int:
        """Names and terms the decoder should favour when the audio is close.

        Each word is boosted token by token inside the decoder (a prefix-trie
        logits bias, `optimized_v18.HotwordBias`); with no words the decode is
        the unchanged one.  Returns how many distinct words are active.
        `strength` is Phonon-1's logit scale; `lam` is Phonon-2's
        (`hotword_lambda`) and is not used by Phonon-1.
        """
        cleaned = []
        for w in (words or ()):
            w = " ".join(str(w).split())
            if w and w not in cleaned:
                cleaned.append(w)
        self.hotwords = cleaned
        if strength is not None:
            self.hotword_strength = float(strength)
        return len(cleaned)

    def _hotword_snapshot(self):
        return (list(self.hotwords), self.hotword_strength)

    def _hotword_restore(self, snap) -> None:
        self.hotwords, self.hotword_strength = list(snap[0]), snap[1]

    def _with_hotwords(self, hotwords, hotword_lambda, fn):
        """Run `fn()` with a per-call hotword list (None = the model's own setting, [] = none for this call) and
        put the model's own setting back afterwards."""
        if hotwords is None and hotword_lambda is None:
            return fn()
        snap = self._hotword_snapshot()
        try:
            self.set_hotwords(self.hotwords if hotwords is None else hotwords, lam=hotword_lambda)
            return fn()
        finally:
            self._hotword_restore(snap)

    def _hotword_kwargs(self) -> dict:
        if not self.hotwords or not getattr(self.model, "hotword_bias", None):
            return {}
        return {"hotwords": list(self.hotwords),
                "hotword_strength": self.hotword_strength}

    @staticmethod
    def _tiered_under_rp() -> bool:
        try:
            from fermion._speech._engine import load as _engine
            return bool(getattr(_engine("optimized_v18"), "TIERED_UNDER_RP", False))
        except Exception:  # pragma: no cover - an older engine copy
            return False

    def describe(self) -> dict:
        """The configuration this process is actually running.

        Printed by `fermion transcribe --verbose` and returned on `/health`,
        for the same reason `serve` names its `--tool-profile`: a benchmark
        must never be silently mis-attributed to a configuration it did not
        use. Every field here is what the decode really did, not what the
        default says it should have.
        """
        return {
            "kind": "speech",
            "profile": self.profile,
            "backend": self.backend,
            "model_dir": str(self.path),
            "audio_tower_bits": self.decode.get("audio_tower_bits"),
            "audio_tower_group_size": self.decode.get("audio_tower_group_size"),
            "fold_bits": self.decode.get("fold_bits"),
            # What the tiered head ACTUALLY does on this path, not what was
            # requested at load.  Since 2026-09-09 (tiered head under the
            # repetition penalty, `optimized_v18.TIERED_UNDER_RP`) the loaded
            # tiered head runs whenever no hotword bias is active; with the
            # switch off it runs only without a penalty, and this engine always
            # decodes with one.  Reporting the configured value regardless would
            # attribute a decode to a path it did not take, which is the one
            # thing this method exists to prevent.
            "tiered_head_bits": (self.decode.get("tiered_head_bits")
                                 if not self.hotwords
                                 and (self._tiered_under_rp()
                                      or self.default_repetition_penalty in (None, 0))
                                 else None),
            "load_seconds": round(self.load_seconds, 3),
            "sample_rate": SAMPLE_RATE,
            "long_audio": _long_audio_note(),
            # The vocabulary bias actually in force for this decode: word
            # count and strength, or none.  Reported for the same reason as
            # every other field here.
            "hotwords": len(self.hotwords),
            "hotword_strength": self.hotword_strength if self.hotwords else None,
        }

    # ------------------------------------------------------------ decoding
    #: The repetition penalty every decode through this engine uses.
    #:
    #: It is here, as a class attribute, so `describe()` can report the decode
    #: path honestly without guessing, and so there is one place to read it.
    #:
    #: It exists for repetition protection: real audio — a live microphone, a
    #: long recording, a pause, silence, noise — can put an autoregressive
    #: decoder into a loop that emits the same phrase until the token budget is
    #: gone. A penalty over a bounded context window is what stops that. The
    #: CLI and the Mac app decode identically for this reason.
    default_repetition_penalty: float = 1.05

    def _decode_single(self, audio, repetition_penalty: float):
        """`stt.py:transcribe_array()` — the graded decode call, for ONE
        window of at most `segment.SINGLE_SHOT_MAX_S` seconds.

        Returns `(text, generation_tokens, token_budget)`; the budget is
        what `truncated` is judged against.
        """
        import mlx.core as mx

        budget = token_budget(len(audio) / SAMPLE_RATE)
        output = self.model.generate(
            audio,
            language="English",
            max_tokens=budget,
            temperature=0.0,
            repetition_penalty=repetition_penalty,
            repetition_context_size=96,
            min_chunk_duration=0.5,
            **self._hotword_kwargs(),
        )
        mx.synchronize()
        text = getattr(output, "text", None)
        if text is None:                                    # pragma: no cover
            text = str(output)
        return text.strip(), int(getattr(output, "generation_tokens", 0) or 0), budget

    def transcribe_array_detailed(self, audio, *,
                                  repetition_penalty: float = 1.05,
                                  hotwords=None, hotword_lambda=None):
        """Decode one array, long or short, into a `segment.Transcript`.

        `hotwords` (a list of names and terms, at most 25 on Phonon-2) and
        `hotword_lambda` (Phonon-2's bonus per matching piece, default 2.0)
        apply to this call only; None keeps the model's `set_hotwords()`
        setting.

        Up to 35 s of audio is one `_decode_single` call, unchanged from
        every earlier release. Longer audio is cut into 25-35 s windows at
        pauses and each window is decoded with that same call and its own
        token budget (see `_speech/segment.py` for the measurements behind
        this). The live session never reaches the segmented path: its
        segments are capped at 30 s.
        """
        from fermion._speech import segment
        from fermion._speech.mlx_memory import clear_mlx_cache

        try:
            return self._with_hotwords(hotwords, hotword_lambda, lambda: segment.transcribe_windows(
                audio, lambda window: self._decode_single(window,
                                                          repetition_penalty)))
        finally:
            clear_mlx_cache(synchronize=True)

    def transcribe_array(self, audio, *, repetition_penalty: float = 1.05,
                         hotwords=None, hotword_lambda=None):
        """(text, decode_seconds, audio_seconds) — the historical shape."""
        return self.transcribe_array_detailed(
            audio, repetition_penalty=repetition_penalty, hotwords=hotwords,
            hotword_lambda=hotword_lambda).triple()

    def transcribe_detailed(self, audio_path, *,
                            repetition_penalty: float = 1.05,
                            hotwords=None, hotword_lambda=None):
        """Transcribe an audio FILE into a `segment.Transcript`."""
        return self.transcribe_array_detailed(
            read_audio(audio_path), repetition_penalty=repetition_penalty,
            hotwords=hotwords, hotword_lambda=hotword_lambda)

    def transcribe(self, audio_path, *, repetition_penalty: float = 1.05,
                   hotwords=None, hotword_lambda=None):
        """Transcribe an audio FILE. Returns (text, decode_seconds, duration).

        `fermion.load_speech("phonon-2").transcribe("call.wav",
        hotwords=["Kushal", "Sigil"])` favours those words for this call."""
        return self.transcribe_detailed(
            audio_path, repetition_penalty=repetition_penalty,
            hotwords=hotwords, hotword_lambda=hotword_lambda).triple()


# --------------------------------------------------------------- audio I/O
def read_audio(path) -> "object":
    """Mono float32 at 16 kHz — `stt.py:transcribe_file()`'s front half.

    `soundfile` reads anything libsndfile does (wav/flac/ogg/aiff/…), which is
    the same set the published CLI accepts. It does NOT read mp3/m4a, and the
    error below says so rather than letting libsndfile's own message
    ("Format not recognised") be the whole explanation.
    """
    import numpy as np
    import soundfile as sf

    path = Path(path).expanduser()
    try:
        if not path.is_file():
            raise SystemExit(f"{path}: no such file")
        if not os.access(path, os.R_OK):
            raise PermissionError(errno.EACCES, "not readable", str(path))
    except PermissionError:
        import getpass
        who = f"user {getpass.getuser()} (uid {os.getuid()})" if hasattr(os, "getuid") else f"user {getpass.getuser()}"
        raise SystemExit(f"{path}: not readable by {who}. Inside a container the audio directory and file must be readable "
                         f"by the container's unprivileged user (chmod a+rX), or run the container with --user $(id -u).")
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except PermissionError:
        raise SystemExit(f"{path}: permission denied reading the file")
    except Exception as exc:                                # noqa: BLE001
        raise SystemExit(
            f"{path}: could not read this audio ({exc}). Supported formats are "
            f"the ones libsndfile handles — wav, flac, ogg, aiff. For mp3/m4a, "
            f"convert first:  ffmpeg -i {path.name} -ar 16000 -ac 1 out.wav")
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # scipy is imported only when a file actually needs resampling, so a
        # 16 kHz file decodes even where scipy is absent.
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return audio


def read_audio_bytes(blob: bytes, filename: str = "upload"):
    """Same, from an in-memory upload (the multipart `file` field)."""
    import io

    import numpy as np
    import soundfile as sf

    try:
        audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    except Exception as exc:                                # noqa: BLE001
        raise ValueError(
            f"could not decode {filename!r} ({exc}). Supported formats are wav, "
            f"flac, ogg and aiff; convert mp3/m4a first "
            f"(ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav).")
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return audio


# ----------------------------------------------------------------- loading
def load(model_dir, *, profile: str, backend: str, quiet: bool = False
         ) -> SpeechModel:
    """Load a profile at the app's shipping decode configuration."""
    from fermion import _compat
    from fermion._speech import decode_kwargs
    from fermion._speech._engine import load as _engine

    _compat.prepare_torch_stack()       # shields first; the runtime imports transformers
    if backend == "phonon2-five-value":
        from fermion._speech import engine_phonon2
        return engine_phonon2.load(model_dir, profile=profile, backend=backend, quiet=quiet)
    kw = decode_kwargs(backend)
    started = time.perf_counter()
    optimized = _engine("optimized_v18")
    model = optimized.load_optimized_v18(Path(model_dir), **kw)
    elapsed = time.perf_counter() - started
    if not quiet:
        bits = kw.get("audio_tower_bits")
        tower = (f"{bits}-bit audio tower (group "
                 f"{kw['audio_tower_group_size']})" if bits else
                 "bf16 audio tower (PHONON_ENC disabled)")
        print(f"[{BRAND}] loaded {profile} in {elapsed:.1f}s — backend "
              f"{backend}, {tower}: the configuration the Mac app ships",
              file=sys.stderr)
    return SpeechModel(model, path=model_dir, profile=profile, backend=backend,
                       decode=kw, load_seconds=elapsed)
