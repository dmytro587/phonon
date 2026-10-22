"""`fermion transcribe` — one-shot speech-to-text.

GRAMMAR: this is `generate`'s twin, not `chat`'s.

`generate` is the one-shot, scriptable, pipeable, byte-reproducible surface;
`chat` is the conversational one. Transcription is one-shot and scriptable by
nature — an audio file in, a line of text out, suitable for a pipe — so it
mirrors `generate`: a positional argument, `--model`, deterministic decoding,
and nothing on stdout but the answer. Every note goes to stderr, so
`fermion transcribe clip.wav > out.txt` writes exactly the transcript.

Determinism is not a choice here: the decode runs at `temperature=0.0`. There
is no `--temperature`, because there is no configuration that uses one.

THERE IS NO `--profile` FLAG, AND THAT IS THE POINT
----------------------------------------------------
Each Phonon model is its own repo, so the model IS the profile:

    FermionResearch/Phonon-1         audio6   415 MB   (the default)
    FermionResearch/Phonon-1-Big     parity   581 MB
    FermionResearch/Phonon-1-Micro   micro    285 MB

A `--profile` flag on top of `--model` would exist only to let someone ask one
repo for a sibling's archive — a 404 at best, and a silently wrong model at
worst. `--model` alone cannot express that mistake. (`fetch._cross_check()`
still checks the repo's own `config.json` agrees about which profile it holds,
because a client-side rule is not a substitute for verifying the artifact.)

REGISTERED THROUGH `_tools.py`, DISPATCHED EARLY — for the same reason as
`models`, and the same reason `serve` has always had an early return in
`cli.main()`: `_tools.dispatch()` runs after `resolve_model(args.model)`, and
`resolve_model` on a repo id downloads a TRTC container. A speech model is not
a TRTC container; letting it reach `resolve_model` would download a Neutrino
8B before failing to find a `.bin`. This command owns its own model lifecycle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fermion import _catalog
from fermion._brand import BRAND, SPEECH_MODEL_ID, DEFAULT_SPEECH_MODEL_ID


def add_parser(sub, common) -> None:
    """`common` is accepted and deliberately UNUSED — see `models.add_parser`.

    The sampler and backend flags `_common_parser()` builds (`--temperature`,
    `--kv-dtype`, `--yarn-factor`, `--backend native|torch`, …) are all
    properties of the TRTC decode path. None of them exist for the MLX speech
    path, and a help page advertising flags that silently do nothing is exactly
    the class of defect `--kv-dtype int8` was made to hard-error on. So this
    parser declares its own small, honest surface. `set_defaults()` is NEVER
    called on `common` here (see `cli._common_parser()`'s docstring).
    """
    p = sub.add_parser("transcribe",
                       help="transcribe an audio file (one-shot, "
                            "deterministic; Apple silicon, and Linux, Windows and Mac CPUs incl. Intel Macs)")
    p.add_argument("words", nargs="*", metavar="MODEL AUDIO",
                   help="the model (phonon-2, phonon-1, phonon-1-big, phonon-1-micro, a repo id or a local model "
                        "directory) followed by the audio file (wav/flac/ogg/aiff); `fermion transcribe phonon-2 meeting.wav`")
    p.add_argument("--model", default=None, help=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true",
                   help="emit a JSON object instead of bare text (text, "
                        "timings, per-segment timestamps, word timestamps, truncated flag)")
    p.add_argument("--verbose", action="store_true",
                   help="print the decode configuration and timings to stderr "
                        "(decode-only and wall-clock, separately)")
    p.add_argument("--threads", type=int, default=None, metavar="N",
                   help="CPU engine threads (default: one per physical core "
                        "from six cores up, every logical cpu on smaller "
                        "parts, performance cores on Apple silicon); same as "
                        "FERMION_CPU_THREADS")
    p.add_argument("--download-only", action="store_true",
                   help="fetch and verify the model, then stop without "
                        "decoding")
    add_hotword_args(p)


def add_hotword_args(p) -> None:
    """`--hotwords`, `--hotwords-file`, `--hotword-lambda` (Phonon-2), `--hotword-strength` (Phonon-1); shared by
    `transcribe` and `listen`."""
    p.add_argument("--hotwords", action="append", default=None, metavar="WORDS",
                   help="names and terms to favour, comma-separated (repeatable; `@FILE` reads a file): people, "
                        "products, places the decoder tends to misspell. Each word is nudged piece by piece inside "
                        "the decoder; the audio still decides. Up to 25 words on Phonon-2. Default: none (the decode "
                        "is unchanged)")
    p.add_argument("--hotwords-file", action="append", default=None, metavar="FILE",
                   help="read hotwords from a file: one term per line, or comma-separated (repeatable)")
    p.add_argument("--hotword-lambda", type=float, default=None, metavar="LAMBDA",
                   help="Phonon-2: the bonus per matching word piece on the decoder's scores (default 2.0); higher "
                        "pulls harder")
    p.add_argument("--hotword-strength", type=float, default=4.0,
                   metavar="LOGITS",
                   help="Phonon-1: how hard --hotwords pull, in logit units (default 4; "
                        "measured: 4 recovers most misspelt names with no "
                        "measurable change to ordinary words, higher values "
                        "start inserting listed words)")


def _resolve(spec: str):
    """(repo_or_None, profile_key, pin, local_dir_or_None) for a --model spec.

    Three things a user may legitimately name:

      1. a published repo id or one of its aliases -> the catalog;
      2. a LOCAL DIRECTORY holding an unpacked profile. This is how a developer
         runs a build that is not published, and it is the only way to run the
         retained `compact` profile, which has no repo at all. The profile is
         identified from the directory name when we recognise it, and is
         otherwise reported as "local";
      3. anything else -> an error that lists what does work.
    """
    from fermion._speech import fetch

    path = Path(str(spec)).expanduser()
    if path.is_dir():
        if not fetch.is_model_dir(path):
            raise SystemExit(
                f"{path} is a directory but does not hold an unpacked Phonon "
                f"model (it needs config.json and packed_manifest.json "
                f"side by side).")
        for key, pin in _catalog.SPEECH_PROFILES.items():
            if path.name == pin["unpack_dir"]:
                return None, key, pin, path
        # A Phonon-2 container directory under any name: its manifest says so.
        try:
            import json as _json
            fmt = _json.load(open(path / "packed_manifest.json")).get("format", "")
        except Exception:                                    # noqa: BLE001
            fmt = ""
        if str(fmt).startswith("fermion-five-value"):
            pin = _catalog.SPEECH_PROFILES["five-value"]
            return None, "five-value", pin, path
        return None, "local", {"backend": "local", "unpack_dir": path.name}, path

    repo = _catalog.resolve_alias(spec)
    entry = _catalog.entry_for(repo)
    if entry is None:
        raise SystemExit(
            f"unknown speech model {spec!r}. Published speech models:\n"
            + "".join(f"    {r}\n" for r in _catalog.SPEECH_REPOS)
            + f"Or pass a local directory holding an unpacked model. "
              f"`{BRAND} models` lists everything.")
    if entry["kind"] != "speech":
        raise SystemExit(
            f"{repo} is a language model, not a speech model. Use "
            f"`{BRAND} generate` for it, or `{BRAND} models` to see what "
            f"speech models exist.")
    key = entry["profile"]
    return repo, key, _catalog.SPEECH_PROFILES[key], None


def _read_hotword_file(path: str) -> list[str]:
    """One term per line, or comma-separated terms on a line; `#` starts a comment."""
    f = Path(path).expanduser()
    try:
        text = f.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"{f}: cannot read the hotwords file ({exc.strerror or exc})")
    out: list[str] = []
    for line in text.splitlines():
        for piece in line.split("#", 1)[0].split(","):
            piece = " ".join(piece.split())
            if piece and piece not in out:
                out.append(piece)
    return out


def parse_hotwords(values, files=None) -> list[str]:
    """`--hotwords "Kushal, Sigil" --hotwords Neutrino --hotwords @names.txt --hotwords-file more.txt` -> the distinct
    words, in order."""
    out: list[str] = []

    def add(piece):
        piece = " ".join(str(piece).split())
        if piece and piece not in out:
            out.append(piece)
    for chunk in values or ():
        chunk = str(chunk)
        if chunk.startswith("@") and len(chunk) > 1:
            for w in _read_hotword_file(chunk[1:]):
                add(w)
            continue
        for piece in chunk.split(","):
            add(piece)
    for f in files or ():
        for w in _read_hotword_file(f):
            add(w)
    return out


def apply_hotwords(speech, args, engine_kind: str) -> None:
    """Attach `--hotwords` to a loaded speech model, or say why not."""
    words = parse_hotwords(getattr(args, "hotwords", None), getattr(args, "hotwords_file", None))
    lam = getattr(args, "hotword_lambda", None)
    if not words:
        return
    if not hasattr(speech, "set_hotwords"):
        print(f"[{BRAND}] note: --hotwords is not available for this model; decoding without it", file=sys.stderr)
        return
    if hasattr(speech, "hotword_lambda"):                      # Phonon-2: every engine (MLX, CPU)
        try:
            n = speech.set_hotwords(words, lam=lam)
        except ValueError as exc:
            raise SystemExit(f"--hotword-lambda: {exc}")
        if getattr(args, "verbose", False):
            print(f"[{BRAND}] {n} hotword{'s' if n != 1 else ''} active, lambda {speech.hotword_lambda:g}",
                  file=sys.stderr)
        return
    if engine_kind != "mlx":
        print(f"[{BRAND}] note: --hotwords for Phonon-1 is available on the Apple-silicon "
              f"(MLX) engine only; decoding without it", file=sys.stderr)
        return
    n = speech.set_hotwords(words, getattr(args, "hotword_strength", None))
    if getattr(args, "verbose", False):
        print(f"[{BRAND}] {n} hotword{'s' if n != 1 else ''} active at strength "
              f"{speech.hotword_strength:g}", file=sys.stderr)


def _take_args(args) -> None:
    """MODEL AUDIO positionals (or --model + AUDIO) -> args.model / args.audio; anything else = usage, exit 2."""
    from fermion import _modelarg
    words = list(getattr(args, "words", None) or [])
    example = "fermion transcribe phonon-2 meeting.wav"
    if args.model:
        if len(words) != 1:
            _modelarg.usage("transcribe", example)
        args.audio = words[0]
        if not os.environ.get("FERMION_QUIET_DEPRECATIONS"):
            print(f"[fermion] note: `--model {args.model}` still works in this release; the form is now "
                  f"`fermion transcribe {args.model} {args.audio}` (the model comes first).", file=sys.stderr)
    else:
        if len(words) != 2:
            _modelarg.usage("transcribe", example)
        args.model_pos, args.audio = words
        _modelarg.take_model(args, "transcribe", example)


def run_early(args) -> int:
    import time
    _take_args(args)
    if getattr(args, "threads", None):
        os.environ["FERMION_CPU_THREADS"] = str(int(args.threads))

    from fermion._speech import backends

    # Wall clock from here: what a user's stopwatch sees for the command,
    # minus only the interpreter start and the argparse above. Reported next
    # to the decode-only time so the two are never confused.
    t_wall = time.perf_counter()
    # 0.2.6: the audio path is checked BEFORE the model is resolved, downloaded or loaded (0.2.5 found a missing file
    # only after the model load, so a cold machine downloaded 164 MB to report a typo).
    if not args.download_only and not Path(args.audio).expanduser().is_file():
        raise SystemExit(f"{args.audio}: no such file")
    repo, key, pin, local_dir = _resolve(args.model)

    # Engine + runtime gate FIRST, before any download: an unsupported
    # machine must be told in one line, not after a 581 MB transfer. This is
    # the explicit-ask path — the user typed a speech verb — so it is a hard
    # refusal, exactly like `--backend native requested but unavailable` in
    # native.try_load(). Which engine (MLX on Apple silicon, CPU on Linux
    # Linux or by FERMION_DEVICE=cpu) is `backends.resolve`'s one decision.
    engine_kind = backends.resolve("`fermion transcribe`")
    backends.require_engine_for(engine_kind, pin.get("backend", ""))

    from fermion._speech import fetch
    model_dir = local_dir if local_dir is not None \
        else fetch.ensure(repo, key, pin)
    if args.download_only:
        print(model_dir)
        return 0

    speech = backends.load(engine_kind, model_dir, profile=key,
                           backend=pin["backend"], quiet=not args.verbose)
    apply_hotwords(speech, args, engine_kind)
    result = speech.transcribe_detailed(args.audio)
    text, decode_s, duration_s = result.triple()
    wall_s = time.perf_counter() - t_wall

    if args.verbose:
        cfg = speech.describe()
        rtf = (duration_s / decode_s) if decode_s > 0 else None
        print(f"[{BRAND}] {json.dumps(cfg)}", file=sys.stderr)
        # Two times, labelled: the decode alone (the number the published
        # realtime factor describes) and the whole command including model
        # load. Through 0.1.22 only the first was printed, and on a long
        # file it flattered a truncated output.
        n_seg = len(result.segments)
        print(f"[{BRAND}] {duration_s:.2f}s audio: decode {decode_s:.2f}s"
              + (f" ({rtf:.1f}x real time, decode only)" if rtf else "")
              + f" · wall {wall_s:.2f}s including model load"
              + f" · {n_seg} segment{'s' if n_seg != 1 else ''}"
              + (" · TRUNCATED: a segment used its whole token budget"
                 if result.truncated else ""),
              file=sys.stderr)
    elif result.truncated:
        print(f"[{BRAND}] note: the output hit its token budget, so part of "
              f"the audio may be missing from the transcript", file=sys.stderr)

    if args.json:
        print(json.dumps({"text": text,
                          "model": repo or str(model_dir),
                          "profile": key,
                          "backend": pin["backend"],
                          "engine": engine_kind,
                          "duration_seconds": round(duration_s, 3),
                          "decode_seconds": round(decode_s, 3),
                          "wall_seconds": round(wall_s, 3),
                          "segment_count": len(result.segments),
                          "segments": result.segments,
                          # 0.2.7: one {text, start, end} per word, seconds from the start of the file, from the
                          # decoder's own token timings. null (not []) when the model's decoder has no timings
                          # (Phonon-1), so "no words" and "no timing available" cannot be confused.
                          "words": result.words if result.timed else None,
                          "truncated": bool(result.truncated)}))
    else:
        print(text)
    return 0


def run(args, kind, path) -> int:                           # pragma: no cover
    """Unreachable — `transcribe` is in `_tools.EARLY`. See `models.run`."""
    print(f"[{BRAND}] internal error: `transcribe` reached the late dispatch "
          f"path", file=sys.stderr)
    return 1
