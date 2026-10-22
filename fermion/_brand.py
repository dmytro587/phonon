"""BRAND GREP POINT 2/3: every user-facing name/id lives here, nowhere else.

NAME FINAL 2026-07-24: this file, pyproject.toml, and the package directory
name were edited, then the wheel rebuilt. Nothing else in the package
hardcodes the brand.
MODEL FAMILY = Neutrino (2026-07-25):
pip/CLI stays `fermion` (the tool is the lab's; the model is what it
serves); DEFAULT_MODEL_ID moved to the one published SKU Neutrino-8B.
"""

BRAND = "fermion"                      # CLI name, import-package name
# PyPI DISTRIBUTION name -- differs from BRAND because bare `fermion` was
# already claimed on PyPI (pyproject.toml [project].name has the full story).
# Anything that prints a `pip install ...` repro line must use THIS, not
# BRAND: `pip install fermion` fails with "No matching distribution found".
PIP_DIST = "fermion-research"
LAB = "fermionresearch"                # HF account (GitHub account is fermionresearch too)
DEFAULT_MODEL_ID = "fermionresearch/Neutrino-8B"  # HF repo id
TOKENIZER_FALLBACK = "Qwen/Qwen3-8B"   # flagship donor tokenizer (Apache-2.0)
def _version() -> str:
    """The installed distribution's version, so `fermion --version` CANNOT
    drift from pyproject.toml again — the 0.1.16 wheel shipped with a stale
    literal here still saying 0.1.15. The literal below is only the fallback
    for a source checkout pip has never seen; keep it equal to pyproject's
    `[project] version` anyway, because the fallback is what a `git clone`
    user sees."""
    try:
        from importlib.metadata import version
        return version(PIP_DIST)
    except Exception:                                       # noqa: BLE001
        return "0.2.9"


VERSION = _version()

# SPEECH (added for the Phonon lane). WHY THE IDS ARE HERE AND THE FACTS ARE
# NOT: this file is the grep point for every user-facing NAME, and a model id
# is a name. Byte counts, SHA-256 pins, archive filenames and unpack
# directories are not names — they are release FACTS with their own upstream
# source of truth (each model repo's published `config.json`), and a
# self-test asserts the two stay in agreement.
# Putting a sha256 in here would make _brand.py a second place release facts
# can drift; putting the repo id in _catalog.py would make _catalog.py a second
# place the brand can drift. So: NAMES here, FACTS in _catalog.py.
# THREE SPEECH REPOS, ONE PER PROFILE (2026-08-26), the way Neutrino ships as
# three. The repo IS the profile: there is no in-repo profile choice, so a
# `--profile` flag would only be a way to ask one repo for another repo's
# archive. Each repo's own `config.json` carries a single `artifact` block.
#
# WHICH ONE IS THE FLAGSHIP (2026-08-28): the `audio6` profile, published as
# `Phonon-1`. It is statistically tied with the larger `parity` build on
# accuracy (paired bootstrap P=0.814 over 5,559 utterances) at 71 % of the
# download (415,077,202 B vs 580,892,956 B) and ~55 % of the disk, and the
# Detta app already defaults to the `audio6` backend. The unqualified name
# therefore resolves to audio6; `parity` keeps shipping as `Phonon-1 Big`
# under its own explicit name.
#
# HUMAN NAMES TAKE A SPACE, IDS TAKE A HYPHEN: "Phonon-1 Big" is what a person
# reads; `Phonon-1-Big` is what a repo id / filename / slug spells.
SPEECH_MODEL_ID = "FermionResearch/Phonon-1"              # flagship (audio6)
SPEECH_BIG_MODEL_ID = "FermionResearch/Phonon-1-Big"      # parity
SPEECH_MICRO_MODEL_ID = "FermionResearch/Phonon-1-Micro"  # micro
SPEECH_MODEL_NAME = "Phonon-1"                    # what a human sees
SPEECH_BIG_MODEL_NAME = "Phonon-1 Big"
SPEECH_MICRO_MODEL_NAME = "Phonon-1 Micro"
SPEECH2_MODEL_ID = "FermionResearch/Phonon-2"             # five-value Parakeet-TDT (staged 2026-09-24)
SPEECH2_MODEL_NAME = "Phonon-2"
# THE DEFAULT SPEECH MODEL (founder, 2026-09-28): a plain `fermion transcribe` / `listen` / `speech bench` / `serve` with no
# --model, and the bare aliases `phonon`, `speech`, `stt`, `asr`, all mean Phonon-2. Phonon-1 stays one `--model phonon-1` away.
DEFAULT_SPEECH_MODEL_ID = SPEECH2_MODEL_ID
SPEECH_FAMILY = "Phonon"                          # the speech model family
LLM_FAMILY = "Neutrino"                           # the language-model family
# The two other published Neutrino repos, named here for `fermion models`.
DRAFT_MODEL_ID = "fermionresearch/Neutrino-0.6B"
CHAT_0P6B_MODEL_ID = "fermionresearch/Neutrino-0.6B-Chat"


# --- FV5 GGUF provenance (2026-09-19) ---------------------------------------
# ggml type ids 43/44 are Fermion FV5/FV5B in our fork builds, but they are NOT
# allocated upstream: the next type merged into llama.cpp takes 43. A file from
# another producer carrying that id would be decoded as FV5 -- silently, as
# garbage. Packs we publish from now on carry a provenance key so a guarded
# build can tell the difference; packs published BEFORE that do not, which is
# why the escape hatch exists rather than a bare refusal.
GGUF_PROVENANCE_KEY = "general.quantization_provenance"
GGUF_PROVENANCE_PREFIX = "fermion-"
LEGACY_GGUF_MESSAGE = (
    "This GGUF was published before Neutrino models carried a provenance "
    "marker, so it cannot be told apart from a file made by someone else that "
    "happens to use the same weight-type numbers. Download the current "
    "gguf/ files, or pass --allow-legacy-gguf to keep using this one."
)
