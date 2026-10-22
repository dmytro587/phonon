"""The lab's model registry — what `fermion` publishes, LLM and speech alike.

WHY THIS FILE EXISTS
--------------------
Until now the package had NO catalog. Model acquisition was a side effect of
`--model <hf repo id>` inside `cli.resolve_model()`, the only friendly-name
mapping anywhere was `draft.CANONICAL_DRAFT`, and the only id a user never had
to type was `_brand.DEFAULT_MODEL_ID`. That is fine for one SKU. It stops being
fine the moment `fermion` serves two model *families* that need different
runtimes, different download shapes and different endpoints — because then
"which model is this?" has to be answerable BEFORE anything is downloaded.

ONE REPO PER SPEECH PROFILE
---------------------------
Speech ships as THREE repos, one per profile, the way Neutrino ships as three.
So a profile determines **which repo to fetch from**, not merely which file
inside one. That is why `repo` is a field of each profile here and why the CLI
has **no `--profile` flag**: the model IS the profile, and a `--profile` flag
would exist only to let someone ask one repo for a sibling's archive, which
404s at best. `--model FermionResearch/Phonon-1-Big` is the whole
interface, and `fermion models` lists three real models.

WHICH REPO IS THE FLAGSHIP (2026-08-28)
---------------------------------------
`Phonon-1` — the `audio6` profile — is the default, and the bare aliases
(`phonon`, `speech`, `stt`, `asr`) resolve to it. It is statistically tied with
the larger `parity` build on accuracy (paired bootstrap P=0.814 over 5,559
utterances) while being 71 % of the download (415,077,202 B vs 580,892,956 B)
and ~55 % of the disk, and the Detta app already ships `audio6` as its backend
default. `parity` still ships, as `Phonon-1 Big`, but only under its own
explicit name. The published names now agree with what actually ships.

REGISTRY STRUCTURE vs BRAND CONSTANTS — the split, and why
----------------------------------------------------------
`fermion/_brand.py` is "BRAND GREP POINT 2/3: every user-facing name/id lives
here, nowhere else". A model id IS a user-facing name, so every repo id in this
file is imported from `_brand`, never typed here.

Everything else in this file is NOT a name. Archive filenames, byte counts and
SHA-256 pins are release FACTS. They have their own upstream source of truth —
each repo's own published `config.json` — and
`fermion/_speech/selftest.py` asserts this file agrees with all three. Putting
a SHA-256 into `_brand.py` would make the brand file a second place a release
fact can drift; putting a repo id in here would make this file a second place
the brand can drift. So: NAMES in `_brand.py`, FACTS here, and a test across
the seam.

THE PINS ARE AUTHORITATIVE, THE HUB'S COPY IS A CROSS-CHECK
-----------------------------------------------------------
The client's own pinned SHA-256 stays authoritative. A repo's published
`config.json` is a cross-check, never an override: a client that finds a
disagreement must refuse the download and tell the user to update, not adopt
the value published there. `fermion/_speech/fetch.py` implements exactly that.

PUBLISHED vs RETAINED
---------------------
Three profiles are PUBLISHED, one per repo. `compact` is RETAINED and has **no
repo at all** — it was cut before publication. It stays in this table so that a
machine which already has it unpacked keeps working and so `fermion models
--all` can explain where it went; it can never be downloaded, because there is
nowhere to download it from. It was cut because it is strictly dominated:
453,985,875 B is 38.9 MB LARGER than Phonon-1's 415,077,202 B, at a worse
measured error rate (3.888105 % vs 3.849609 % on the 400-utterance common set).
"""
from __future__ import annotations

from fermion._brand import (
    CHAT_0P6B_MODEL_ID,
    DEFAULT_MODEL_ID,
    DRAFT_MODEL_ID,
    LLM_FAMILY,
    SPEECH_BIG_MODEL_ID,
    SPEECH_BIG_MODEL_NAME,
    SPEECH_FAMILY,
    SPEECH_MICRO_MODEL_ID,
    SPEECH_MICRO_MODEL_NAME,
    SPEECH_MODEL_ID,
    SPEECH_MODEL_NAME,
    SPEECH2_MODEL_ID,
    SPEECH2_MODEL_NAME,
)

# --------------------------------------------------------------- speech facts
# Source of truth: each repo's published config.json (its `artifact`
# block). Byte counts and digests are the roundtrip-verified values from
# the release receipts (2026-08-24) and are immutable for this release.
#
# `unpack_dir` is the ONE field here with no upstream any more: the published
# config.json files stopped carrying it when the repos were split, and with the
# public script tree cancelled there is no longer a second unpacker whose
# directory names we must match. It is therefore a CLI-owned convention. The
# historical names are kept anyway, so a directory this CLI produces is still
# interchangeable with one the app's developer tree or an older quickstart
# produced.

SPEECH_PROFILES = {
    # The DEFAULT is first, because `published_speech_profiles()` is the order
    # a user is shown and the flagship should lead.
    "audio6": {
        "repo": SPEECH_MODEL_ID,
        "name": SPEECH_MODEL_NAME,
        "character": "Recommended default; the Big build's measured accuracy "
                     "at 71 % of the download",
        "backend": "audio6",
        "filename": "phonon-audio6.bps.tar.zst",
        "sha256": "214c3b45aa57257013811a53f99a905848466ad4ab21f2b2f8368f7ac79427b2",
        "download_bytes": 415_077_202,
        "unpacked_bytes": 454_527_730,
        "unpack_dir": "model_v18_mlx_head8audio6_quint5",
        "published": True,
    },
    "parity": {
        "repo": SPEECH_BIG_MODEL_ID,
        "name": SPEECH_BIG_MODEL_NAME,
        "character": "Largest build; the quickest response",
        "backend": "parity-fold4-tiered4",
        "filename": "phonon-parity.bps.tar.zst",
        "sha256": "5dbd566d022a05bac045d1e84f0bf6188999494070e2a00dc5fd1bac6caffdb9",
        "download_bytes": 580_892_956,
        "unpacked_bytes": 822_143_951,
        "unpack_dir": "model_v18_mlx_quint5",
        "published": True,
    },
    "micro": {
        "repo": SPEECH_MICRO_MODEL_ID,
        "name": SPEECH_MICRO_MODEL_NAME,
        "character": "Smallest install; trades some accuracy for size",
        "backend": "micro",
        "filename": "phonon-micro.bps.tar.zst",
        "sha256": "8a03e55d78e65c10f6ce22aff7e32359ca6a7aaa27973435b9ebc3cdbff13f9c",
        "download_bytes": 285_083_584,
        "unpacked_bytes": 331_234_013,
        "unpack_dir": "model_v18_mlx_hybrid4_quint5",
        "published": True,
    },
    # STAGED 2026-09-24, NOT PUBLISHED YET — Phonon-2, the five-value Parakeet-TDT
    # 0.6B. One container file (`model.fermion`) + the mlx-audio config; the
    # runtime is `fermion._speech.engine_phonon2` over `_engine_phonon2/`.
    # `published` flips to True on the founder's word, together with the
    # selftest's "exactly three published" expectation and the repo going public.
    "five-value": {
        "repo": SPEECH2_MODEL_ID,
        "name": SPEECH2_MODEL_NAME,
        "character": "Five-value Parakeet-TDT; the smallest download",
        "backend": "phonon2-five-value",
        "filename": "phonon-2.bps.tar.zst",
        "sha256": "98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e",
        "download_bytes": 163_515_201,
        "unpacked_bytes": 177_716_675,
        "unpack_dir": "model_phonon2_c4c_int6",
        "published": True,
        "staged": True,
    },
    # RETAINED, NEVER PUBLISHED — `repo` is None ON PURPOSE. `fetch.ensure()`
    # refuses rather than guessing a URL, and nothing in the CLI can construct
    # a download for it. Only a directory already on disk can run it.
    "compact": {
        "repo": None,
        "name": "Compact",
        "character": "Cut before release (larger than Phonon-1 and less "
                     "accurate)",
        "backend": "compact-fold4",
        "filename": "phonon-compact.bps.tar.zst",
        "sha256": "72735a14d7fbf2dca114abbf1fe75a21921d505e59e435c913423bb096925131",
        "download_bytes": 453_985_875,
        "unpacked_bytes": 500_030_084,
        "unpack_dir": "model_v18_mlx_hybrid8_quint5",
        "published": False,
    },
}

# FOUNDER CALL (staged 2026-09-26, both variants prepared; A is live): what `fermion transcribe` runs with no --model and
# what the bare aliases (`phonon`, `speech`, `stt`, `asr`) resolve to.
#   A) "audio6"     Phonon-1 stays the default; Phonon-2 is reached by name (`--model phonon-2`).
#   B) "five-value" Phonon-2 becomes the default; Phonon-1 stays one `--model phonon-1` away.
# Flipping is this one line plus the alias block in MODELS (see SPEECH_DEFAULT_IS_PHONON2). Nothing else moves.
DEFAULT_SPEECH_PROFILE = "five-value"   # FOUNDER 2026-09-28: Phonon-2 is the default; bare phonon/speech/stt/asr resolve to it
SPEECH_DEFAULT_IS_PHONON2 = DEFAULT_SPEECH_PROFILE == "five-value"

#: repo id -> profile key. The reverse lookup the CLI actually uses: a user
#: names a MODEL, and the model determines the profile.
SPEECH_REPOS = {p["repo"]: key
                for key, p in SPEECH_PROFILES.items() if p["repo"]}


def published_speech_profiles() -> list[str]:
    """Profile keys a user is offered, in the order they should be shown."""
    return [k for k, v in SPEECH_PROFILES.items() if v["published"]]


def profile_for_repo(repo: str) -> tuple[str, dict] | None:
    """(profile_key, pin) for a speech repo id, or None."""
    key = SPEECH_REPOS.get(repo)
    return (key, SPEECH_PROFILES[key]) if key else None


# ---------------------------------------------------------------- the models
# One entry per repo the lab publishes. `kind` decides EVERYTHING downstream:
# which runtime loads it, which download path fetches it, and which endpoints
# `fermion serve` mounts.
#
#   kind="llm"     TRTC v4 container; cli.resolve_model + the native/torch
#                  backends; /v1/chat/completions + /v1/completions.
#   kind="speech"  Phonon byte-plane archive; fermion._speech; MLX on Apple
#                  silicon only; /v1/audio/transcriptions.

MODELS = {
    DEFAULT_MODEL_ID: {
        "kind": "llm",
        "family": LLM_FAMILY,
        "name": "Neutrino-8B",
        "summary": "An 8B language model for chat and generation (`fermion chat neutrino-8b`).",
        "aliases": ("neutrino", "neutrino-8b", "8b"),
        "default": True,
    },
    DRAFT_MODEL_ID: {
        "kind": "llm",
        "family": LLM_FAMILY,
        "name": "Neutrino-0.6B",
        "summary": "A 0.6B model that speeds up Neutrino-8B as its draft with --draft auto.",
        "aliases": ("neutrino-0.6b", "0.6b", "draft"),
        "default": False,
    },
    CHAT_0P6B_MODEL_ID: {
        "kind": "llm",
        "family": LLM_FAMILY,
        "name": "Neutrino-0.6B-Chat",
        "summary": "A 0.6B model tuned for conversation.",
        "aliases": ("neutrino-0.6b-chat", "0.6b-chat"),
        "default": False,
    },
    SPEECH2_MODEL_ID: {
        "kind": "speech",
        "family": SPEECH_FAMILY,
        "name": SPEECH2_MODEL_NAME,
        "summary": "The current speech model, a 164 MB download (`fermion transcribe phonon-2 audio.wav`).",
        "aliases": (("phonon-2", "phonon2", "phonon", "speech", "stt", "asr") if SPEECH_DEFAULT_IS_PHONON2 else ("phonon-2", "phonon2")),
        "default": False,
        "profile": "five-value",
    },
    SPEECH_MODEL_ID: {
        "kind": "speech",
        "family": SPEECH_FAMILY,
        "name": SPEECH_MODEL_NAME,
        "summary": "The earlier speech model, a 415 MB download.",
        # `phonon` and the bare family aliases point at the FLAGSHIP, so
        # `fermion transcribe` with no --model does the expected thing.
        "aliases": (("phonon-1",) if SPEECH_DEFAULT_IS_PHONON2 else ("phonon", "phonon-1", "speech", "stt", "asr")),
        "default": False,
        "profile": "audio6",
    },
    SPEECH_BIG_MODEL_ID: {
        "kind": "speech",
        "family": SPEECH_FAMILY,
        "name": SPEECH_BIG_MODEL_NAME,
        "summary": "The largest Phonon build.",
        "aliases": ("phonon-1-big", "phonon-big", "big"),
        "default": False,
        "profile": "parity",
    },
    SPEECH_MICRO_MODEL_ID: {
        "kind": "speech",
        "family": SPEECH_FAMILY,
        "name": SPEECH_MICRO_MODEL_NAME,
        "summary": "The smallest Phonon model, for the smallest install. It "
                   "trades some accuracy for size.",
        "aliases": ("phonon-1-micro", "phonon-micro", "micro"),
        "default": False,
        "profile": "micro",
    },
}

# alias -> canonical repo id. Follows the `draft.CANONICAL_DRAFT` precedent
# (a short name the user may type resolves to one real repo id) and extends it
# to every SKU instead of just the draft.
ALIASES = {alias: repo
           for repo, entry in MODELS.items()
           for alias in entry["aliases"]}

# The repo ids themselves, case-folded, so `--model fermionresearch/phonon-1`
# reaches the same model as `--model FermionResearch/Phonon-1`. The Hub is
# case-insensitive about repo ids and users type what they read; the CANONICAL
# spelling is still what we send.
ALIASES.update({repo.lower(): repo for repo in MODELS})


def resolve_alias(spec: str) -> str:
    """Map an alias or a differently-cased repo id to its canonical repo id.

    Anything unknown passes through untouched, so a local path or a
    third-party repo id still means exactly what it meant in 0.1.14.
    """
    if not spec:
        return spec
    return ALIASES.get(str(spec).strip().lower(), spec)


def entry_for(spec: str) -> dict | None:
    """The catalog entry for a spec (repo id or alias), or None if unknown."""
    return MODELS.get(resolve_alias(spec))


def kind_of(spec: str) -> str:
    """"llm" | "speech" | "unknown" — WITHOUT touching the network or disk.

    This is the function `fermion serve` uses to decide which endpoints to
    mount, so it must never download anything to answer.

    An unknown spec is "unknown", not "llm": a local path or a third-party repo
    id is a perfectly legitimate thing to hand `--model`, and the caller
    decides what to do with it (serve treats unknown as llm, which is the
    pre-existing behaviour and keeps every 0.1.14 invocation working).
    """
    entry = entry_for(spec)
    if entry is not None:
        return entry["kind"]
    return "unknown"


def is_speech(spec: str) -> bool:
    return kind_of(spec) == "speech"
