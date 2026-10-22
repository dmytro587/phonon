"""`fermion models` — what the lab publishes, and what is already on disk.

REGISTERED THROUGH `_tools.py`, DISPATCHED EARLY. Why both:

`_tools.register()` is pure argparse, so registering here is free and keeps
`cli.main()` to the same two small insertions it already had. But
`_tools.dispatch()` runs AFTER `resolve_model(args.model)` in `main()`, and
`resolve_model` on a repo id DOWNLOADS THE MODEL. A command whose entire job is
to tell you which models exist and which you have must not download an 8B
container to answer, so `models` uses the early-dispatch hook — the same
mechanism `serve` has always used (`if args.cmd == "serve": … return`), for the
same reason: it owns its own model lifecycle.

This command NEVER touches the network. Everything it prints comes from
`fermion._catalog` (static) and from the local cache (a directory test).
"""
from __future__ import annotations

import json
import sys

from fermion._brand import BRAND, PIP_DIST
from fermion import _catalog


def add_parser(sub, common) -> None:
    """`common` is accepted and deliberately UNUSED.

    `_common_parser()` is a factory of sampler/backend flags for a decode
    command. `models` decodes nothing: inheriting `--temperature`, `--kv-dtype`
    and the rest would print a help page full of flags that do nothing. It is
    also the reason this file must never call `set_defaults()` on `common` —
    see the load-bearing docstring at `cli._common_parser()`.
    """
    p = sub.add_parser("models",
                       help="list the models the lab publishes and show which "
                            "are already downloaded")
    p.add_argument("--json", action="store_true",
                   help="machine-readable output")
    p.add_argument("--all", action="store_true",
                   help="also show profiles that are retained but no longer "
                        "published")


# ------------------------------------------------------------------ helpers
def _mb(n: int) -> str:
    return f"{n / 1e6:,.0f} MB"


def _speech_row(repo: str, entry: dict) -> dict:
    """One speech MODEL. Each repo holds exactly one profile, so a model row
    and a profile row are the same thing now."""
    from fermion._speech import fetch
    key = entry["profile"]
    pin = _catalog.SPEECH_PROFILES[key]
    return {
        "model": repo,
        "name": entry["name"],
        "summary": entry["summary"],
        "aliases": list(entry["aliases"]),
        "profile": key,
        "backend": pin["backend"],
        "download_bytes": pin["download_bytes"],
        "unpacked_bytes": pin["unpacked_bytes"],
        "default": repo == _catalog.resolve_alias("phonon"),
        "installed": fetch.installed(repo, pin["unpack_dir"]),
        "path": str(fetch.profile_dir(repo, pin["unpack_dir"])),
    }


def _retained_rows() -> list[dict]:
    """Profiles that exist in the catalog but were never published."""
    return [{"profile": key, "name": pin["name"],
             "character": pin["character"],
             "download_bytes": pin["download_bytes"]}
            for key, pin in _catalog.SPEECH_PROFILES.items()
            if not pin["published"]]


def _llm_installed(repo: str) -> bool:
    """Is this LLM repo already in the Hugging Face cache? No network.

    Deliberately a cache-layout test rather than `snapshot_download(...,
    local_files_only=True)`: that call constructs a repo object and can still
    touch the network in some hub versions, and this command promises not to.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:                                       # noqa: BLE001
        return False
    from pathlib import Path

    import os
    root = Path(os.environ.get("HF_HUB_CACHE") or HF_HUB_CACHE)
    d = root / ("models--" + repo.replace("/", "--"))
    if not d.is_dir():
        return False
    snaps = d / "snapshots"
    if not snaps.is_dir():
        return False
    return any(any(rev.iterdir()) for rev in snaps.iterdir() if rev.is_dir())


def collect(*, show_all: bool = False) -> dict:
    from fermion._speech import backends
    out = {"llm": [], "speech": [], "speech_support": backends.describe()}
    for repo, entry in _catalog.MODELS.items():
        if entry["kind"] == "llm":
            out["llm"].append({
                "model": repo,
                "name": entry["name"],
                "summary": entry["summary"],
                "aliases": list(entry["aliases"]),
                "default": entry["default"],
                "installed": _llm_installed(repo),
            })
        else:
            out["speech"].append(_speech_row(repo, entry))
    if show_all:
        out["retained"] = _retained_rows()
    return out


# --------------------------------------------------------------------- run
def run_early(args) -> int:
    data = collect(show_all=getattr(args, "all", False))
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2))
        return 0

    print("Models published by Fermion Research\n")

    print("LANGUAGE  (fermion chat / generate / serve)")
    for row in data["llm"]:
        mark = "*" if row["installed"] else " "
        print(f"  {mark} {row['model']}")
        print(f"      {row['summary']}")
        print(f"      aliases: {', '.join(row['aliases'])}")
    print()

    print("SPEECH    (fermion transcribe / serve)")
    for row in data["speech"]:
        mark = "*" if row["installed"] else " "
        print(f"  {mark} {row['model']}   {_mb(row['download_bytes'])}")
        print(f"      {row['summary']}")
        print(f"      aliases: {', '.join(row['aliases'])}")
    print()
    for row in data.get("retained", []):
        print(f"    ({row['name']}: {row['character']} — "
              f"{_mb(row['download_bytes'])}, no repo)")
    if data.get("retained"):
        print()

    support = data["speech_support"]
    if support["available"]:
        engine = support.get("engine")
        print(f"  speech runtime: ready on {support['this_platform']}"
              + (f" ({engine} engine)" if engine else ""))
        if support.get("note"):
            print(f"      note: {support['note']}")
    else:
        print(f"  speech runtime: NOT available here — {support['why_not']}")
    print()
    print("  * = already on this machine")
    print(f"  install:    pip install {PIP_DIST}")
    print(f"  chat:       {BRAND} chat neutrino-8b")
    print(f"  transcribe: {BRAND} transcribe phonon-2 audio.wav")
    print(f"  serve:      {BRAND} serve phonon-2")
    if not getattr(args, "all", False):
        print(f"  ({BRAND} models --all also lists unreleased builds)")
    return 0


def run(args, kind, path) -> int:                           # pragma: no cover
    """Present only so this module satisfies the `_tools` module contract.

    It is unreachable: `models` is in `_tools.EARLY`, so `dispatch_early()`
    handles it before `resolve_model()` ever runs. If it were ever reached, a
    model would already have been downloaded to answer a question that needs no
    model, so it fails loudly rather than pretending.
    """
    print(f"[{BRAND}] internal error: `models` reached the late dispatch path",
          file=sys.stderr)
    return 1
