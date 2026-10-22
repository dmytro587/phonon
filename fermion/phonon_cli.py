"""`phonon`, `phonon-2`, `phonon-1`: the fermion CLI with the model fixed (founder's call, 2026-09-28/30).

    phonon transcribe meeting.wav      == fermion transcribe phonon-2 meeting.wav
    phonon listen                      == fermion listen phonon-2
    phonon serve [--port ...]          == fermion serve phonon-2 ...
    phonon bench --audio clip.wav      == fermion speech bench phonon-2 --audio clip.wav
    phonon describe  (or --describe)   == fermion describe        (CPU features + kernel tier, 0.2.5)
`phonon` is Phonon-2; `phonon-2` and `phonon-1` name their model in the command."""
from __future__ import annotations

import sys

from fermion._brand import BRAND

VERBS = ("transcribe", "listen", "serve", "bench", "describe")


def _run(model: str, prog: str) -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "--describe":
        argv = ["describe", *argv[1:]]
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in VERBS:
        print(f"{prog}: the {BRAND} CLI with the model fixed to {model}.\n\n"
              f"  {prog} transcribe meeting.wav\n  {prog} listen\n  {prog} serve [--port 8000] [--unix-socket PATH]\n"
              f"  {prog} bench --audio clip.wav\n  {prog} describe\n\nEverything after the verb is passed to `{BRAND} <verb> {model}`; "
              f"`{prog} <verb> -h` shows that verb's options.", file=sys.stderr)
        raise SystemExit(2 if not argv or argv[0] not in ("-h", "--help") else 0)
    verb, rest = argv[0], argv[1:]
    if verb == "bench":
        sys.argv = [BRAND, "speech", "bench", model, *rest]
    elif verb == "describe":
        sys.argv = [BRAND, "describe", *rest]
    else:
        sys.argv = [BRAND, verb, model, *rest]
    from fermion.cli import main
    main()


def main_phonon() -> None:
    _run("phonon-2", "phonon")


def main_phonon2() -> None:
    _run("phonon-2", "phonon-2")


def main_phonon1() -> None:
    _run("phonon-1", "phonon-1")
