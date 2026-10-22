"""Speech support for the `fermion` CLI — Phonon on Apple silicon via MLX.

THE FAST-PATH PIN (the whole point of this module)
--------------------------------------------------
The CLI must default to the SAME shipping-fast configuration the Mac app
uses, not a slow reference path. Anyone benchmarking from pip must measure
the fast path, because that is what the published numbers describe.

That is the same defect `fermion/native.py` was written to fix on the LLM side:
through 0.1.3 the CLI downloaded the fast runtime and then never executed it,
so `pip install fermion && fermion chat` was two orders of magnitude slower
than the number on the model card. The published numbers were binary numbers;
the shipped CLI was not using the binary. This module exists so the same thing
cannot happen to speech.

WHAT THE APP ACTUALLY SHIPS — verified, not assumed
----------------------------------------------------
Two settings, from two different places in the shipping Mac app (Detta):

  1. `PHONON_ENC=q8` — the app defaults the 8-bit audio tower ON
     (gated 2026-08-26: all accuracy gates held at-or-better; buys
     -14.7% time-to-first-word, -9% median latency, -175 MB).
     The engine parses that value into `audio_tower_bits=8,
     audio_tower_group_size=128` — the regex is `q(\\d+)(?:g(\\d+))?` with
     the group size defaulting to 128.

  2. backend `parity-fold4-tiered4` — the `parity` profile's backend name,
     which the engine expands to `fold_bits=4, tiered_head_bits=4`, with
     `tiered_head_k` keeping the loader default of 32.

So the app's effective decode configuration is exactly the five values in
`DECODE` below, and `fermion/_speech/selftest.py` re-derives all five and
asserts equality, so the app and the CLI cannot silently drift apart.

NOTE ON `PHONON_ENC` ITSELF: it keeps its name on purpose (it is read by the
MODEL's runtime, and the model is still called Phonon; only the application
is named Detta). We honour the same variable here, so a user who exports
`PHONON_ENC=` to turn the 8-bit tower OFF gets the same behaviour from the
CLI as from the app.

WHY THE RUNTIME IS VENDORED IN THE WHEEL
-----------------------------------------
It is 112 KB of pure Python, it keeps the wheel `py3-none-any`, it rides
`pip install -U` like every other CLI fix, and it does not touch the
published model repositories.
"""
from __future__ import annotations

import os
import re

# --------------------------------------------------------------- the pin
# The app's shipping decode configuration, as loader keyword arguments.
# CHANGE NOTHING HERE WITHOUT CHANGING THE APP AND RE-RUNNING THE SELFTEST.
DECODE = {
    "fold_bits": 4,            # parity-fold4-tiered4 -> fold4
    "tiered_head_bits": 4,     # parity-fold4-tiered4 -> tiered4
    "tiered_head_k": 32,       # loader default; the app never overrides it
    "audio_tower_bits": 8,     # PHONON_ENC=q8
    "audio_tower_group_size": 128,   # PHONON_ENC=q8 -> group default 128
}

#: The env var the app sets, and its shipped value. Honoured here too, so the
#: CLI and the app respond identically to a user who overrides it.
ENC_ENV = "PHONON_ENC"
ENC_SHIPPED = "q8"

#: The engine backend name the fold/tiered build runs — the `parity` profile,
#: published as Phonon-1 Big. This is the backend the DECODE pin above
#: describes, so it is the one the pin test measures.
#:
#: NOT the default profile any more: since 2026-08-28 the default is `audio6`
#: (Phonon-1), whose backend name is literally "audio6" and which runs with no
#: fold/tiered stage. `decode_kwargs()` below maps every backend, so the
#: default path is covered too; this constant stays pinned to the fold/tiered
#: shape because that is the shape the DECODE dict spells out.
SHIP_BACKEND = "parity-fold4-tiered4"

_ENC_RE = re.compile(r"q(\d+)(?:g(\d+))?")


def _parse_enc(value: str):
    """The engine's env-lever parse, reimplemented to the same regex.

    Returns (bits, group_size), or None for "no quantization requested", or
    raises ValueError for a malformed value. The app PRINTS and ignores a
    malformed value; we do the same at the call site.
    """
    value = (value or "").strip().lower()
    if not value:
        return None
    m = _ENC_RE.fullmatch(value)
    if not m:
        raise ValueError(value)
    return int(m.group(1)), int(m.group(2) or 128)


def decode_kwargs(profile_backend: str = SHIP_BACKEND) -> dict:
    """The loader kwargs for one profile backend, honouring $PHONON_ENC.

    This is THE function the pin test measures, and the only place the CLI
    decides how to decode.

    * `parity-fold4-tiered4` (Phonon-1 Big) -> fold4 + tiered4.
    * `audio6` / `micro` / `compact-fold4` -> their own shapes. Their artifacts
      already carry a quantized audio tower, so the 8-bit tower lever is a
      NO-OP on them by construction (`v18_runtime.quantize_audio_tower` skips
      every `nn.QuantizedLinear` it meets) — it is still passed, exactly as the
      app passes it, so the two code paths stay identical rather than merely
      equivalent.
    """
    kw = dict(DECODE)
    # $PHONON_ENC overrides the shipped q8, including "off" (empty string).
    raw = os.environ.get(ENC_ENV)
    if raw is not None:
        try:
            parsed = _parse_enc(raw)
        except ValueError:
            import sys
            print(f"[phonon] ignoring invalid {ENC_ENV}={raw!r}",
                  file=sys.stderr)
            parsed = (DECODE["audio_tower_bits"],
                      DECODE["audio_tower_group_size"])
        if parsed is None:
            kw["audio_tower_bits"] = None          # explicitly disabled
            kw["audio_tower_group_size"] = 128
        else:
            kw["audio_tower_bits"], kw["audio_tower_group_size"] = parsed
    # Fold/tiered come from the profile's backend name, exactly as the
    # engine's backend table maps them.
    if profile_backend == "parity-fold4-tiered4":
        kw["fold_bits"], kw["tiered_head_bits"] = 4, 4
    elif profile_backend == "compact-fold4":
        kw["fold_bits"], kw["tiered_head_bits"] = 4, None
    elif profile_backend in ("audio6", "micro"):
        kw["fold_bits"], kw["tiered_head_bits"] = None, None
    else:                       # unknown backend: run the plain profile path
        kw["fold_bits"], kw["tiered_head_bits"] = None, None
    return kw
