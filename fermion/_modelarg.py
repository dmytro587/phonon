"""0.2.4 (founder's call): the `fermion` CLI never guesses a model.

`fermion serve <model>`, `fermion transcribe <model> <audio>` and `fermion listen <model>` take the model as the first
positional argument (`--model <model>` means the same thing); a bare verb prints this usage and exits 2 — no download,
no default. The alias table is the catalog's."""
from __future__ import annotations

import os
import sys

from fermion import _catalog
from fermion._brand import BRAND

_ORDER = ["FermionResearch/Phonon-2", "FermionResearch/Phonon-1", "FermionResearch/Phonon-1-Big",
          "FermionResearch/Phonon-1-Micro", "fermionresearch/Neutrino-8B", "fermionresearch/Neutrino-0.6B",
          "fermionresearch/Neutrino-0.6B-Chat"]


def model_lines() -> list[str]:
    out = []
    for repo in _ORDER + [r for r in _catalog.MODELS if r not in _ORDER]:
        e = _catalog.MODELS.get(repo)
        if not e:
            continue
        kind = "speech" if _catalog.is_speech(repo) else "language"
        aliases = ", ".join(e.get("aliases", [])[:3])
        out.append(f"  {aliases:34s} {kind:8s} {repo}")
    return out


def usage(verb: str, example: str) -> "NoReturn":  # noqa: F821
    msg = [f"{BRAND} {verb}: name the model. Phonon never guesses.", "",
           f"  {example}", "", "Models (alias, kind, repo):", *model_lines(), "",
           f"`{BRAND} models` lists everything and marks what is already on this machine."]
    print("\n".join(msg), file=sys.stderr)
    raise SystemExit(2)


def take_model(args, verb: str, example: str, positional_attr: str = "model_pos") -> str:
    """Merge the positional model and --model; refuse (usage, exit 2) when neither is given or they disagree."""
    pos = getattr(args, positional_attr, None)
    opt = getattr(args, "model", None)
    if pos and opt and _catalog.resolve_alias(pos) != _catalog.resolve_alias(opt):
        print(f"{BRAND} {verb}: `{pos}` and --model `{opt}` name different models; pass one.", file=sys.stderr)
        raise SystemExit(2)
    model = opt or pos
    if not model:
        usage(verb, example)
    if opt and not pos and not os.environ.get("FERMION_QUIET_DEPRECATIONS"):
        print(f"[{BRAND}] note: `--model {opt}` still works in this release; the form is now `{BRAND} {verb} {opt} ...` "
              f"(the model comes first).", file=sys.stderr)
    args.model = model
    return model
