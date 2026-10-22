"""`fermion listen` — live streaming microphone speech-to-text.

GRAMMAR: this is `transcribe`'s live twin. `transcribe` is audio file in, one
line of text out; `listen` is the microphone in, the same text out — printed
LIVE while the user speaks (a partial hypothesis overwriting one terminal
line, finalized segments printed permanently), and, on Ctrl-C, the full
transcript on stdout. stdout carries NOTHING ELSE: every live line, note and
timing goes to stderr, exactly `transcribe`'s pipe discipline, so
`fermion listen > note.txt` captures exactly the words that were spoken.

`--wav FILE` streams a file through the IDENTICAL code path, paced to real
time. It exists so the whole streaming stack — segmentation, partial cadence,
final decode, rendering — is testable without a microphone; mic capture is
the only layer a machine cannot exercise. See `fermion/_speech/live.py` for
the one session both feeds share, and for why the decode configuration is
`transcribe`'s, unchanged.

`--model` semantics are `transcribe`'s, byte for byte (it calls the same
resolver), and THERE IS NO `--profile` FLAG for the same reason `transcribe`
has none: each Phonon model is its own repo, so the model IS the profile.

REGISTERED THROUGH `_tools.py`, DISPATCHED EARLY — for the same reason as
`transcribe` (see its module docstring): this command owns its own model
lifecycle, and letting a speech repo id reach `resolve_model()` would download
a Neutrino container before failing to find a `.bin` in it.

MICROPHONE CAPTURE is `sounddevice` (a 16 kHz mono float32 `InputStream`),
probed by a listen-specific gate (`gate.require_listen`) so `transcribe` and
`serve` acquire no new requirement because a sibling verb grew one. The
documented install line does not change: `mlx-audio` already depends on
`sounddevice>=0.5.3`, so the line the README already prints installs it.
"""
from __future__ import annotations

import argparse
import sys
import time

from fermion import _catalog
from fermion._brand import BRAND, SPEECH_MODEL_ID, DEFAULT_SPEECH_MODEL_ID


def add_parser(sub, common) -> None:
    """`common` is accepted and deliberately UNUSED — see `transcribe.add_parser`."""
    p = sub.add_parser("listen",
                       help="live microphone transcription — speak, watch the "
                            "text; Ctrl-C stops (Apple silicon)")
    p.add_argument("--wav", default=None, metavar="FILE",
                   help="stream this audio file through the identical live "
                        "path, paced to real time, instead of capturing the "
                        "microphone (the whole streaming stack, headless)")
    p.add_argument("model_pos", nargs="?", metavar="MODEL",
                   help="the speech model: phonon-2, phonon-1, phonon-1-big, phonon-1-micro, a repo id or a local model "
                        "directory; `fermion listen phonon-2`")
    p.add_argument("--model", default=None, help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true",
                   help="print the decode configuration and one timed stderr "
                        "line per partial/final instead of the live display")
    from fermion.transcribe import add_hotword_args
    add_hotword_args(p)


class _Renderer:
    """The live stderr display. stdout is reserved for the final transcript.

    Two modes, chosen once:

      * ANIMATED (stderr is a tty, not --verbose): the current partial
        overwrites one line via `\\r`, truncated to the terminal width showing
        the TAIL (the words being spoken now); a final clears that line and
        prints permanently.
      * LOGGED (--verbose, or stderr is not a tty): one plain timed line per
        event, so a redirected stderr shows the live cadence instead of a
        carriage-return soup. Partials are logged only under --verbose;
        finals always print.
    """

    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.animate = sys.stderr.isatty() and not verbose
        self.partials = 0
        self._shown = 0                      # width of the animated line
        self._t0 = time.monotonic()

    def _clear(self) -> None:
        if self._shown:
            sys.stderr.write("\r" + " " * self._shown + "\r")
            self._shown = 0

    def partial(self, text: str) -> None:
        self.partials += 1
        if self.animate:
            import shutil
            width = max(shutil.get_terminal_size().columns - 1, 8)
            line = text if len(text) <= width else "…" + text[-(width - 1):]
            pad = " " * max(self._shown - len(line), 0)
            sys.stderr.write("\r" + line + pad)
            sys.stderr.flush()
            self._shown = len(line)
        elif self.verbose:
            print(f"[{BRAND}] partial {time.monotonic() - self._t0:7.2f}s  "
                  f"{text}", file=sys.stderr)

    def final(self, text: str) -> None:
        self._clear()
        if self.verbose:
            print(f"[{BRAND}] final   {time.monotonic() - self._t0:7.2f}s  "
                  f"{text}", file=sys.stderr)
        else:
            print(text, file=sys.stderr)
        sys.stderr.flush()


def run_early(args) -> int:
    from fermion._speech import gate
    from fermion import _modelarg
    _modelarg.take_model(args, "listen", "fermion listen phonon-2")

    from fermion.transcribe import _resolve
    repo, key, pin, local_dir = _resolve(args.model)

    # Platform + runtime gate FIRST, before any download — `transcribe`'s
    # rule. Mic mode additionally needs `sounddevice`, probed here and ONLY
    # here: `--wav` mode reads a file, so it keeps `transcribe`'s exact gate.
    if args.wav:
        gate.require("`fermion listen`")
    else:
        gate.require_listen("`fermion listen`")

    from fermion._speech import backends, fetch, live
    model_dir = local_dir if local_dir is not None \
        else fetch.ensure(repo, key, pin)
    # Through `backends.load` for the one always-on engine banner; the
    # engine is the MLX one (gated above), exactly as before.
    speech = backends.load("mlx", model_dir, profile=key,
                           backend=pin["backend"], quiet=not args.verbose)
    from fermion.transcribe import apply_hotwords
    apply_hotwords(speech, args, "mlx")
    warm_s = live.warm_up(speech)   # Metal compile now, not on the 1st partial
    if args.verbose:
        print(f"[{BRAND}] warm-up decode {warm_s:.1f}s — paid here so the "
              f"first partial lands on cadence", file=sys.stderr)

    ui = _Renderer(args.verbose)
    session = live.LiveSession(speech, on_partial=ui.partial,
                               on_final=ui.final)
    try:
        if args.wav:
            if args.verbose:
                print(f"[{BRAND}] streaming {args.wav} at real time — the "
                      f"microphone path minus the microphone", file=sys.stderr)
            transcript = live.run_wav(session, args.wav)
        else:
            print(f"[{BRAND}] listening — speak; Ctrl-C stops and prints the "
                  f"transcript", file=sys.stderr)
            transcript = live.run_mic(session)
    except KeyboardInterrupt:
        # Ctrl-C outside run_mic's own handler (e.g. mid-`--wav`, or during a
        # decode): finalize what is buffered; a second Ctrl-C keeps only what
        # already finalized. Never a traceback.
        try:
            transcript = session.finish()
        except KeyboardInterrupt:
            transcript = session.transcript
    ui._clear()
    print(transcript)
    return 0


def run(args, kind, path) -> int:                           # pragma: no cover
    """Unreachable — `listen` is in `_tools.EARLY`. See `models.run`."""
    print(f"[{BRAND}] internal error: `listen` reached the late dispatch "
          f"path", file=sys.stderr)
    return 1
