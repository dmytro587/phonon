"""Phonon-2 on MLX: the five-value Parakeet-TDT container through mlx-audio's graph.

Same `SpeechModel` contract as `engine.py` (transcribe / transcribe_detailed /
transcribe_array_detailed / describe), so every entry point -- `transcribe`,
`serve`, `listen`, `bench` -- works unchanged. Differences from Phonon-1:

  * the 264 encoder linears run as two native 2-bit affine QMMs each
    (`PackedFiveValueLinear`), never expanded; everything else is bf16;
  * decoding is greedy TDT (mlx-audio), no token budget and no repetition
    penalty exist in this decoder -- the arguments are accepted and ignored;
  * audio longer than `segment.SINGLE_SHOT_MAX_S` is still cut at pauses by
    `segment.transcribe_windows`, exactly as for Phonon-1, so the long-audio
    behaviour of the CLI is one code path for both models.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fermion._brand import BRAND
from fermion._speech.engine import SAMPLE_RATE, SpeechModel
from fermion._speech.hotwords import Phonon2HotwordsMixin

CONTAINER = "model.fermion"
CONFIG = "config.json"


class Phonon2SpeechModel(Phonon2HotwordsMixin, SpeechModel):
    kind = "speech"

    def describe(self) -> dict:
        return {
            "kind": "speech",
            "profile": self.profile,
            "backend": self.backend,
            "model_dir": str(self.path),
            "runtime": "mlx-audio parakeet graph; five-value encoder linears as 2-bit planes",
            "activations": "bfloat16",
            "five_value_modules": self.decode.get("five_value_modules"),
            "levers": self.decode.get("levers"),
            "levers_error": self.decode.get("levers_error"),
            "load_seconds": round(self.load_seconds, 2),
            **self._hotword_describe(),
        }

    def _generate_biased(self, x, automaton):
        """model.generate with the hotword-armed TDT loop (fast_decode.set_hotwords + `_decode_fast`), whatever loop
        is installed as `model.decode`; the model is put back exactly as it was afterwards."""
        import mlx.core as mx
        from fermion._speech._engine_phonon2 import load as _engine
        fd = _engine("fast_decode")
        m = self.model
        had = "decode" in vars(m)
        prev = vars(m).get("decode")
        fd.set_hotwords(m, automaton)
        m.decode = fd.biased_decode(m)
        try:
            return m.generate(x, dtype=mx.bfloat16)
        finally:
            fd.set_hotwords(m, None)
            if had:
                m.decode = prev
            else:
                del m.decode

    def _decode_single(self, audio, repetition_penalty: float):
        import mlx.core as mx
        import numpy as np
        from fermion._speech.mlx_memory import clear_mlx_cache

        x = mx.array(np.asarray(audio, dtype=np.float32))
        automaton = self._hotword_automaton(self.model.vocabulary) if self.hotwords else None
        try:
            if automaton is None:
                out = self.model.generate(x, dtype=mx.bfloat16)
            else:
                out = self._generate_biased(x, automaton)
            mx.synchronize()
            text = getattr(out, "text", "") or ""
            # 0.2.7: the decoder's own token timings (start frame and duration of every emitted piece, as the
            # graph returns them) travel with the text so `segment.transcribe_windows` can build word timestamps.
            # The text itself is unchanged: it is still `out.text`, never rebuilt from the tokens.
            tokens = [(tok.text, float(tok.start), float(tok.duration))
                      for sent in (getattr(out, "sentences", []) or []) for tok in sent.tokens]
            # No token budget exists on a transducer; report the frame count as the
            # budget so `truncated` (tokens >= budget) can never trigger.
            return text.strip(), len(tokens), int(len(audio) / SAMPLE_RATE * 12.5) + 1_000_000, tokens
        finally:
            clear_mlx_cache()


def load(model_dir, *, profile: str, backend: str, quiet: bool = False) -> Phonon2SpeechModel:
    from fermion._speech._engine_phonon2 import load as _engine
    model_dir = Path(model_dir)
    container = model_dir / CONTAINER
    config = model_dir / CONFIG
    if not container.is_file() or not config.is_file():
        raise SystemExit(f"{model_dir} does not hold a Phonon-2 model "
                         f"({CONTAINER} and {CONFIG} side by side)")
    started = time.perf_counter()
    lp = _engine("load_parakeet")
    import mlx.core as mx
    receipts: dict = {}
    model = lp.load_parakeet(container, config, dtype=mx.bfloat16, packed=True,
                             scale_dtype=mx.float32, receipts=receipts)
    levers = apply_levers(model, receipts)
    elapsed = time.perf_counter() - started
    if not quiet:
        print(f"[{BRAND}] loaded {profile} in {elapsed:.1f}s — backend {backend}, "
              f"{receipts.get('substituted', 0)} five-value linears as 2-bit planes, "
              f"bf16 elsewhere", file=sys.stderr)
    return Phonon2SpeechModel(model, path=model_dir, profile=profile, backend=backend,
                              decode={"five_value_modules": receipts.get("substituted"), "levers": levers,
                                      "levers_error": receipts.get("levers_error")},
                              load_seconds=elapsed)


#: `FERMION_P2_FAST` -- comma list of post-load speed levers (fast_decode.py). Default `tdt16,dense16`
#: (founder rule 2026-09-28: same accuracy wins): `tdt16` keeps the greedy TDT loop on the GPU (one host sync
#: per 16 steps; tokens/timings/text identical to mlx-audio's loop on 400 utterances + 20 long holds); `dense16`
#: runs the 264 encoder linears as one fp16 GEMM over the exact dequantised weights instead of two 2-bit QMMs
#: (174x vs 109x on an M5 on long audio; +0.4 GB peak; 400-utterance ruler 228 errors on both paths, G-DECODE
#: 40/40, 660-clip gate 7.630 vs 7.653 -- ~2.5 % of utterances differ in wording at the same error rate).
#: `tdt16` alone = the bit-exact path; `""` = the stock loop.
LEVERS_ENV = "FERMION_P2_FAST"
LEVERS_DEFAULT = "tdt16,dense16"


def apply_levers(model, receipts: dict) -> list[str]:
    levers = [x for x in os.environ.get(LEVERS_ENV, LEVERS_DEFAULT).split(",") if x]
    if not levers:
        return []
    try:
        import mlx.core as mx
        from fermion._speech._engine_phonon2 import load as _engine
        fd = _engine("fast_decode")
        receipts["levers"] = {}
        for lv in levers:
            if lv.startswith("tdt"):
                fd.install_fast_tdt(model, block=int(lv[3:] or 16)); receipts["levers"][lv] = "installed"
            elif lv in ("dense16", "dense32"):
                receipts["levers"][lv] = fd.densify_encoder(model, mx.float16 if lv == "dense16" else mx.float32)
            elif lv == "compile":
                receipts["levers"][lv] = fd.install_compiled_layers(model)
            else:
                raise ValueError(f"{LEVERS_ENV}: unknown lever {lv!r}")
        return levers
    except Exception as exc:                      # never lose the decode over a speed lever
        receipts["levers_error"] = f"{type(exc).__name__}: {exc}"
        return []
