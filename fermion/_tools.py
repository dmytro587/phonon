"""Registration hook for the analysis/trust commands (inspect, verify, bench).

Kept in one place so `cli.py` needs exactly two small insertions: `register()`
next to the other subparsers and `dispatch()` right after the model is
resolved. Each command lives in its own module.
"""
from __future__ import annotations

from pathlib import Path

_ORDER = ("inspect", "verify", "bench")

# Commands that OWN THEIR OWN MODEL LIFECYCLE, and so must run BEFORE
# `cli.main()` calls `resolve_model(args.model)`.
#
# `dispatch()` below runs after the model is resolved, which is right for the
# analysis commands: `inspect`/`verify`/`bench` all analyse a resolved
# container, so resolving it first is the work, not a side effect. It is wrong
# for these two:
#
#   models      answers "which models exist and which do I have". Downloading
#               an 8B container to answer that would be absurd; it must not
#               touch the network at all.
#   transcribe  loads a SPEECH model, which is not a TRTC container.
#               `resolve_model` would download Neutrino-8B and then fail to
#               find a `.bin` in it.
#   listen      is `transcribe`'s live twin (microphone / `--wav`), and owns
#               its model lifecycle for exactly the same reason.
#
# `serve` has always solved this with an early return in `cli.main()` (`if
# args.cmd == "serve": … return`). `EARLY` is that same escape hatch, made
# reusable, so the new verbs need one insertion in `main()` between them
# rather than one each. Registration still happens in `register()` below, so
# `--help` and the subparser table are built in exactly one place.
#   describe    (0.2.5) reports the host's CPU features and the speech kernel tier; nothing to resolve or download.
EARLY = ("models", "transcribe", "listen", "speech", "describe")

DRAFT_HELP = ("path to a small TRTC v4 container (the 0.6B) to draft with. "
              "Emitted tokens are verified against the big model: on "
              "CPU/greedy the text is token-identical to a plain run; on "
              "CUDA bf16 near-tie positions can flip — run `fermion verify` "
              "on your pair.")


def _mod(name: str):
    import importlib
    return importlib.import_module(f"fermion.{name}")


def add_draft_flag(parser) -> None:
    """Add --draft to a parser unless it already has one (serve defines its
    own; we never fight it)."""
    if any("--draft" in a.option_strings for a in parser._actions):
        return
    parser.add_argument("--draft", default=None, help=DRAFT_HELP)


def register(sub, common) -> None:
    # `common` may be a parser OR a zero-arg factory that builds one. Prefer
    # the factory: argparse's set_defaults() mutates action.default on the
    # PARENT's Action objects, which parents=[common] shares with every other
    # subparser, so `verify`'s set_defaults(max_new=32) silently rewrote the
    # default for chat/generate/serve too (caught 2026-07-27; see the
    # _common_parser() docstring in cli.py). One fresh parent per subcommand
    # keeps each set_defaults local.
    make = common if callable(common) else (lambda: common)
    for name in _ORDER + EARLY:
        _mod(name).add_parser(sub, make())
    # chat/generate get --draft here rather than in cli.py's `common` block:
    # `serve` already declares its own --draft, so a shared flag would collide.
    for name in ("chat", "generate"):
        if name in sub.choices:
            add_draft_flag(sub.choices[name])


def dispatch(args, kind: str, path: Path):
    """Return an exit code if `args.cmd` is one of ours, else None."""
    if args.cmd not in _ORDER:
        return None
    return _mod(args.cmd).run(args, kind, Path(path))


def dispatch_early(args):
    """Run a command that must not have a model resolved for it.

    Called from `cli.main()` immediately after `parse_args()` and BEFORE
    `resolve_model()`. Returns an exit code, or None when this command is not
    one of ours and normal dispatch should continue.
    """
    if args.cmd not in EARLY:
        return None
    return _mod(args.cmd).run_early(args)
