"""fermion CLI: chat REPL + one-shot generate + OpenAI-compatible server for
TRTC v4 containers and the tiny QR/GIF artifact models.

All inference is banked code. TRTC v4 has TWO decode backends behind one
interface:

  * **native** (default when available) — the prebuilt `fermion-run-<platform>`
    runtime the model repo ships under `bin/`, driven by `fermion/native.py`.
    This is the path every published speed number was measured on.
  * **torch** (fallback + reference) — the hf_ternary loader (correctness-
    gated) + transformers generate.

Tiny artifacts go through the float64 reference decoder that the browser demo
is gated against. This file only routes, times, and prints.

`iter_generate()` below is the SINGLE decode path: `chat`, `generate`,
`bench` and `fermion serve` (fermion/server.py) all go through it, so served
output is token-identical to CLI output for the same prompt, backend and
sampler settings.

BACKEND HISTORY (fixed 2026-07-27):
through 0.1.3 the installer DOWNLOADED `bin/fermion-run-*` (it is in
`HF_RUNTIME_PATTERNS` below) and then never executed it — every token came
out of the torch reference path at ~1 tok/s for the 0.6B, against the 150+
tok/s the same container does through the binary on the same laptop. The
published numbers were binary numbers; the shipped CLI was not using the
binary. `--backend` now selects, and `auto` prefers native.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from fermion._brand import BRAND, DEFAULT_MODEL_ID, TOKENIZER_FALLBACK, VERSION
from fermion._container import TINY_MAGICS, TRTC_MAGIC, read_magic

# Files the runtime actually opens, and nothing else. `snapshot_download`
# with no filter pulls the WHOLE repo at `main`: for Neutrino-8B that is
# 10.56 GB across 59 files to run a 3.89 GB model, because the repo also
# carries a GGUF build (4.09 GB), the tv4z transport (2.56 GB) and a second
# copy of the tokenizer under mlx/. On a home connection that is ~20 extra
# minutes on the very first command a new user types. These patterns are
# matched with fnmatch against the repo-relative path, so an unanchored name
# like "config.json" matches only the root copy — `mlx/tokenizer.json` and
# `gguf/receipts/*.json` do NOT match, which is the whole point.
# Everything the runtime needs EXCEPT the weights themselves. The weights come
# from exactly one of two files chosen at download time (below): the raw `.bin`
# (today's path) or the smaller `.tv4z` transport we decode locally.
HF_WEIGHTLESS_PATTERNS = [
    # the transformers config directory the loader reads
    "config.json",
    "generation_config.json",
    # tokenizer, in every layout transformers/tokenizers may ask for
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    # length + sha256 pins that `fermion info` verifies the download against
    "MANIFEST.json",
    # the native speed-path runner the model card points at, + sha sidecars
    # (1.87 MB, 0.05% of the download; the torch path here never execs it)
    "bin/fermion-run-*",
    # the licence those weights and binaries are used under (11 KB)
    "LICENSE",
]

# The raw-bin fetch: the weightless set + the runnable container itself (also
# the only pattern the tiny QRB8/QRL8 repos and any repo without a `.tv4z`
# match). This is the byte-for-byte pre-0.1.9 download set.
HF_RUNTIME_PATTERNS = HF_WEIGHTLESS_PATTERNS + ["*.bin"]

# SHIPPING sampler config (graded 2026-07-26). The native runtime no-ops
# every sampler flag at temperature 0, which is why
# the graded config runs at temp 0.01 rather than 0. The torch path has no
# such coupling: it can run a DETERMINISTIC argmax with the repetition
# penalty still applied, which is what temperature==0 maps to here.
SHIP_TEMPERATURE = 0.01
SHIP_TOP_P = 1.0
SHIP_REP_PENALTY = 1.05
SHIP_PEN_WINDOW = 256

# THE DEFAULTS SPLIT (ruled 2026-07-27, documented in --help and the README
# because an UNDOCUMENTED version of this split is what caused the argparse
# regression this comment sits next to):
#   chat + serve -> the graded SHIPPING SAMPLED config above. These are the
#     conversational surfaces, and the shipping config is what they were
#     graded at, so a human talking to the model gets the graded behaviour.
#   generate     -> DETERMINISTIC greedy, no penalty. It is the scriptable,
#     pipeable, reproducible path: the banked receipts, the token-identity
#     gates, `fermion verify` and the acceptance harness all depend on
#     `generate` being byte-reproducible at its default. Sampled one-shots
#     are one flag away (`--temperature 0.01 --rep-penalty 1.05`).
CLI_REP_PENALTY = 1.0        # generate's default: penalty off
GENERATE_TEMPERATURE = 0.0   # generate's default: greedy

# ...and the same split applies to the token budget. 0.1.7 raised `chat` from
# 256 to 1024: at 256 the REPL truncated three CORRECT answers mid-conclusion
# in the behavioural deep-dive, which is indistinguishable from the model
# failing. `generate` stays at 256 for the same reproducibility reason as the
# sampler defaults.
CHAT_MAX_NEW = 1024
GENERATE_MAX_NEW = 256

SPLIT_NOTE = (f"`chat` and `serve` default to the graded shipping config "
              f"(temp {SHIP_TEMPERATURE} / top-p {SHIP_TOP_P} / rep-pen "
              f"{SHIP_REP_PENALTY} / window {SHIP_PEN_WINDOW}); `generate` "
              f"stays deterministic (greedy, no penalty) so scripts and "
              f"receipts reproduce.")


# --------------------------------------------------------------- resolution
def _cache_dir() -> Path:
    d = Path.home() / ".cache" / BRAND
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_model(spec: str) -> tuple[str, Path]:
    """Return (kind, path): kind in {trtc-dir, trtc-bin, tiny}."""
    p = Path(spec).expanduser()
    if p.is_dir():
        if (p / "config.json").exists():
            return "trtc-dir", p
        bins = sorted(p.glob("*.bin"))
        for b in bins:
            # four bytes, read as four bytes: read_bytes()[:4] used to pull
            # the whole container into RAM to look at its magic.
            if read_magic(b) == TRTC_MAGIC:
                return "trtc-bin", b
        raise SystemExit(f"{p}: no config.json or TRTC .bin inside")
    if p.is_file():
        magic = read_magic(p)
        if magic == TRTC_MAGIC:
            return "trtc-bin", p
        if magic in TINY_MAGICS:
            return "tiny", p
        raise SystemExit(f"{p}: unknown magic {magic!r} (want TRTC/QRB8/QRL8)")
    # not a local path: treat as an HF repo id. First give the catalog a
    # chance to turn a short alias into one — `--model neutrino-0.6b` ->
    # `fermionresearch/Neutrino-0.6B`. This is the `draft.CANONICAL_DRAFT`
    # precedent (`--draft auto` -> one real repo id) generalised; a spec the
    # catalog does not know passes through untouched, so every 0.1.14
    # invocation still means exactly what it meant.
    from fermion._catalog import is_speech, resolve_alias
    repo = resolve_alias(spec)
    if is_speech(repo):
        # A speech repo carries no TRTC container, so downloading it here
        # would spend hundreds of megabytes to fail. Name the right verb.
        raise SystemExit(
            f"{repo} is a speech model — it has no TRTC container to load "
            f"here. Use `{BRAND} transcribe <audio>` for a one-shot "
            f"transcript, or `{BRAND} serve --model {spec}` for an "
            f"OpenAI-compatible /v1/audio/transcriptions endpoint.")
    if repo != spec:
        _note(f"--model {spec} -> {repo}")
    return resolve_model(_download_hf(repo))


def _note(msg: str) -> None:
    print(f"[{BRAND}] {msg}", file=sys.stderr, flush=True)


def _hf_progress_bars_on() -> None:
    # A cold 8B fetch is ~GBs / minutes of wall time, and it was a BLANK
    # terminal for all of it (measured 2026-07-27 on a clean machine: the only
    # output before the first prompt was an HF token warning). Make sure the
    # hub's tqdm bars are actually on -- transformers or a parent process may
    # have disabled them globally, and a first-run user cannot tell
    # "downloading" from "hung". The env override is respected: if the user set
    # HF_HUB_DISABLE_PROGRESS_BARS themselves, we leave the bars off.
    if not os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"):
        try:
            from huggingface_hub.utils import enable_progress_bars
            enable_progress_bars()
        except Exception:
            pass


# ------------------------------------------------------ disk-space handling
# 0.1.11, after an HN user with 750 GB free hit "No space left on device" on
# the quickstart (reproduced + diagnosed on a clean machine):
# the model cache lives on the HOME volume no matter where the big disk is
# mounted, and neither hf_xet's traceback nor pip's error names the directory
# that filled. Everything below exists to (a) refuse to START a weights
# download the cache volume cannot hold, and (b) turn a mid-download or
# mid-decode ENOSPC into one message that names the path and the fix,
# instead of a traceback.

_SPACE_MARGIN = 64 * 2**20    # insist on this much headroom beyond the bytes


def _cache_override() -> str | None:
    """$FERMION_CACHE_DIR (0.1.11): where this CLI's model downloads land.

    Passed as `cache_dir=` to every snapshot_download call, so it moves the
    model cache without touching the rest of the Hugging Face cache. Setting
    HF_HOME instead also works (and additionally moves the hub's xet staging
    area and the tokenizer-fallback cache).
    """
    d = os.environ.get("FERMION_CACHE_DIR")
    return os.path.expanduser(d) if d else None


def _download_cache_root() -> Path:
    """The directory snapshot_download writes into — the one that can fill."""
    ov = _cache_override()
    if ov:
        return Path(ov)
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _gb(n) -> str:
    return f"{n / 10**9:.2f} GB"


def _is_enospc(exc: BaseException) -> bool:
    """True when `exc` (or its cause/context chain) is a full disk.

    Catches the raw OSError(errno 28), plain "No space left on device" text,
    and hf_xet's `RuntimeError: ... IO Error: No space left on device
    (os error 28)`, which arrives with no .errno at all.
    """
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, OSError) and e.errno == errno.ENOSPC:
            return True
        msg = str(e)
        if "No space left on device" in msg or "os error 28" in msg:
            return True
        e = e.__cause__ or e.__context__
    return False


def _space_fix_lines() -> str:
    """The two-knob fix, identical in every disk-space message."""
    return (f"[{BRAND}]   export HF_HOME=/big/disk/hf                  "
            f"# the standard Hugging Face knob\n"
            f"[{BRAND}]   export FERMION_CACHE_DIR=/big/disk/fermion   "
            f"# this CLI's model downloads only")


def _exit_disk_full(during: str) -> None:
    """A write hit ENOSPC: say WHICH directory filled, then exit cleanly."""
    root = _download_cache_root()
    print(f"[{BRAND}] ERROR: no space left on device while {during}.\n"
          f"[{BRAND}] The volume holding {root} is full —\n"
          f"[{BRAND}] that is where the model cache lives: usually your home "
          f"volume, even when a\n"
          f"[{BRAND}] bigger disk is mounted elsewhere. Free up space there, "
          f"or move the cache to\n"
          f"[{BRAND}] a disk with room and re-run:\n"
          + _space_fix_lines(),
          file=sys.stderr, flush=True)
    raise SystemExit(1)


def _preflight_space(local: Path, spec: str, needed: int,
                     breakdown: str) -> None:
    """Refuse to start a weights download the cache volume cannot hold.

    `local` is the just-fetched metadata snapshot — it is IN the cache, so
    disk_usage on it measures exactly the volume the weights will land on.
    Never blocks on a guess it cannot make: an unreadable path skips the
    check (the ENOSPC catch downstream still protects the download).
    """
    try:
        free = shutil.disk_usage(local).free
    except OSError:
        return
    if free >= needed + _SPACE_MARGIN:
        return
    root = _download_cache_root()
    print(f"[{BRAND}] ERROR: not enough disk space for {spec}.\n"
          f"[{BRAND}]   model cache dir : {root}\n"
          f"[{BRAND}]   space needed    : ~{_gb(needed)} ({breakdown})\n"
          f"[{BRAND}]   space free      : {_gb(free)}\n"
          f"[{BRAND}] No weights were downloaded. Free up space there, or "
          f"move the cache to a disk\n"
          f"[{BRAND}] with room and re-run:\n"
          + _space_fix_lines(),
          file=sys.stderr, flush=True)
    raise SystemExit(1)


def _snapshot(during: str, **kw) -> Path:
    """snapshot_download + the cache override + a human ENOSPC message."""
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(cache_dir=_cache_override(), **kw))
    except Exception as exc:
        if _is_enospc(exc):
            _exit_disk_full(during)
        raise


def _drop_transport(tv4z_path: Path) -> None:
    """Delete the `.tv4z` after a VERIFIED decode (0.1.11).

    The transport is download-time packaging: once the decoded `.bin` has
    matched the MANIFEST sha256, keeping it would cost an 8B user 2.56 GB of
    cache forever (~6.4 GB steady-state to run a 3.88 GB model). Called ONLY
    after the sha gate passes — a failed or unverified decode keeps the
    transport, which is the user's only copy of the bytes they paid bandwidth
    for. FERMION_KEEP_TRANSPORT=1 opts out. In the hub cache the snapshot
    entry is a symlink to a content-addressed blob: both go. This cannot
    cause re-downloads: every later run returns at the cached-bin check
    before the transport is ever requested again (gated). Never fatal — worst case the file stays.
    """
    if os.environ.get("FERMION_KEEP_TRANSPORT"):
        _note(f"keeping the transport {tv4z_path.name} "
              f"(FERMION_KEEP_TRANSPORT is set)")
        return
    try:
        gb = tv4z_path.stat().st_size / 2**30
        blob = tv4z_path.resolve() if tv4z_path.is_symlink() else None
        tv4z_path.unlink()
        if blob is not None and blob.is_file():
            blob.unlink()
        _note(f"removed the {gb:.2f} GB transport {tv4z_path.name} — the "
              f"decoded bin is the runnable copy "
              f"(FERMION_KEEP_TRANSPORT=1 keeps transports)")
    except OSError:
        pass


def _manifest_weights(local: Path):
    """(bin_name, sha256, bytes) from a fetched MANIFEST.json, else Nones."""
    try:
        wc = (json.loads((local / "MANIFEST.json").read_text())
              .get("weights_container") or {})
    except (OSError, ValueError, AttributeError):
        return None, None, None
    return wc.get("file"), wc.get("sha256"), wc.get("bytes")


def _hub_bin_bytes(spec: str) -> int | None:
    """Total bytes of the repo's .bin files, sized off the Hub API.

    The sane-estimate path for repos without a MANIFEST.json (the tiny
    QRB8/QRL8 artifacts). Any API failure returns None — no preflight then;
    the ENOSPC catch stays as the backstop.
    """
    import fnmatch
    try:
        from huggingface_hub import HfApi
        sizes = [f.size for f in HfApi().list_repo_tree(spec, recursive=True)
                 if getattr(f, "size", None)
                 and fnmatch.fnmatch(f.path, "*.bin")]
        return sum(sizes) or None
    except Exception:
        return None


def _tv4z_plan(local: Path, spec: str):
    """Decide the tv4z download+decode plan for a fetched repo, or None.

    Returns `(tv4z_name, tv4z_bytes, bin_name, bin_sha, bin_bytes)` when this
    repo ships a `.tv4z` transport that we can decode locally into the
    MANIFEST-pinned `.bin` — otherwise None, which routes the caller to the
    raw-bin download (the tiny QRB8/QRL8 repos, any repo without a transport,
    or any platform without a decoder). Never raises: a None here is always
    the safe fallback.
    """
    if os.environ.get("FERMION_NO_TV4Z"):
        return None
    from fermion import _transport
    if not _transport.can_decode():
        return None
    mf = local / "MANIFEST.json"
    if not mf.is_file():
        return None
    try:
        doc = json.loads(mf.read_text())
    except (OSError, ValueError):
        return None
    wc = doc.get("weights_container") or {}
    bin_name, bin_sha = wc.get("file"), wc.get("sha256")
    if not bin_name or not bin_sha or not str(bin_name).endswith(".bin"):
        return None
    # the transport file the MANIFEST names, else the bin name with a .tv4z
    # suffix (the published naming convention: neutrino-8b_v4.{bin,tv4z}).
    ct = wc.get("coded_transport") or {}
    tv4z_name = ct.get("file") or (bin_name[:-4] + ".tv4z")
    # One tree listing answers two questions: does the transport exist, and —
    # when the MANIFEST has no coded_transport byte count (the 0.6B repos) —
    # how big it is, so the 0.1.11 disk preflight can count it.
    tv4z_bytes = ct.get("bytes")
    try:
        from huggingface_hub import HfApi
        tree = {f.path: getattr(f, "size", None)
                for f in HfApi().list_repo_tree(spec, recursive=True)}
        if tv4z_name not in tree:
            return None
        if tv4z_bytes is None:
            tv4z_bytes = tree[tv4z_name]
    except Exception:
        return None
    return tv4z_name, tv4z_bytes, bin_name, bin_sha, wc.get("bytes")


def _bin_already_valid(binpath: Path, bin_sha: str, bin_bytes) -> bool:
    """Cheap reuse gate for a `.bin` already on disk (prior version or decode).

    Existence + length + TRTC magic — the sha was proven the one time it was
    decoded/downloaded, so we do NOT re-hash multiple GB on every launch (which
    is exactly the re-work this whole path exists to avoid).
    """
    if not binpath.is_file():
        return False
    if bin_bytes is not None and binpath.stat().st_size != bin_bytes:
        return False
    return read_magic(binpath) == TRTC_MAGIC


def _download_hf(spec: str) -> Path:
    """Fetch an HF repo, preferring the compressed `.tv4z` transport.

    Flow: pull the small metadata first; if the repo ships a decodable `.tv4z`
    and we do not already have a valid `.bin`, pull the transport (2.56 GB for
    the 8B, not 3.88) and decode it in place, verifying the decoded bytes
    against the MANIFEST sha256. Anything that is not a clean win — no
    transport, no decoder, decode error, or a sha mismatch — falls back to the
    raw-bin download, which is byte-for-byte the pre-0.1.9 behaviour. The one
    exception (0.1.11): a decode that failed because the DISK FILLED never
    falls back — the raw bin is a larger file on the same full volume — and
    every weights download is preceded by a free-space preflight against the
    MANIFEST byte counts. Also 0.1.11: the transport is deleted after the
    sha-verified decode (`_drop_transport`), so the PEAK need is transport +
    bin but the steady state is just the bin.
    """
    _note(f"fetching {spec} from the Hugging Face Hub "
          f"(first run downloads the model, ~GBs; cached after)")
    _hf_progress_bars_on()

    if os.environ.get("FERMION_DOWNLOAD_ALL"):
        return _snapshot("downloading the full model repo", repo_id=spec)

    # 1. metadata first (config/tokenizer/MANIFEST/LICENSE/native runner) — a
    #    few MB, and it tells us which weights file to pull.
    local = _snapshot("downloading the model metadata", repo_id=spec,
                      allow_patterns=HF_WEIGHTLESS_PATTERNS)

    plan = _tv4z_plan(local, spec)
    if plan is None:
        # Raw-bin path. Preflight (0.1.11): the MANIFEST pins the weights
        # size; a repo without one (the tiny artifacts) is sized off the Hub
        # API. Skipped when the bin is already on disk (nothing left to
        # download) or nothing is knowable.
        bin_name, bin_sha, bin_bytes = _manifest_weights(local)
        cached = (bin_name is not None
                  and _bin_already_valid(local / bin_name, bin_sha, bin_bytes))
        if not cached:
            need = bin_bytes or _hub_bin_bytes(spec)
            if need:
                _preflight_space(local, spec, need,
                                 f"{_gb(need)} weights download")
        return _snapshot("downloading the model weights", repo_id=spec,
                         allow_patterns=HF_RUNTIME_PATTERNS)

    tv4z_name, tv4z_bytes, bin_name, bin_sha, bin_bytes = plan
    binpath = local / bin_name
    tv4z_path = local / tv4z_name

    # 2. already have the runnable bin (prior version, or a prior decode)?
    if _bin_already_valid(binpath, bin_sha, bin_bytes):
        _note(f"using cached {bin_name} ({binpath.stat().st_size / 2**30:.2f} "
              f"GB already on disk; no re-download or decode)")
        return local

    # 2b. preflight (0.1.11): this volume must hold the transport download
    #     AND the decoded bin. A transport already fully on disk (a prior run
    #     that died mid-decode) only needs the bin's bytes.
    need, parts = 0, []
    have_tv4z = (tv4z_bytes is not None and tv4z_path.is_file()
                 and tv4z_path.stat().st_size == tv4z_bytes)
    if tv4z_bytes and not have_tv4z:
        need += tv4z_bytes
        parts.append(f"{_gb(tv4z_bytes)} transport download")
    if bin_bytes:
        need += bin_bytes
        parts.append(f"{_gb(bin_bytes)} decoded container")
    if need:
        _preflight_space(local, spec, need, " + ".join(parts))

    # 3. pull the transport and decode it in place.
    from fermion import _transport
    from fermion._container import sha256_file
    try:
        local = _snapshot(f"downloading the {tv4z_name} transport",
                          repo_id=spec,
                          allow_patterns=HF_WEIGHTLESS_PATTERNS + [tv4z_name])
        tv4z_path = local / tv4z_name
        z_gb = tv4z_path.stat().st_size / 2**30
        _note(f"decoding the {z_gb:.2f} GB transport {tv4z_name} to "
              f"{bin_name} (one-time, local; runtime is unchanged after)…")
        secs = _transport.decode(tv4z_path, binpath, on_note=None)
        got = sha256_file(binpath)
        if got != bin_sha:
            raise _transport.DecodeError(
                f"decoded sha256 {got[:16]}… != MANIFEST pin {bin_sha[:16]}… "
                f"— the bytes are not the bytes we published")
        b_gb = binpath.stat().st_size / 2**30
        _note(f"decoded {bin_name} in {secs:.1f}s, byte-identical to the "
              f"MANIFEST sha256 ({b_gb:.2f} GB on disk from a {z_gb:.2f} GB "
              f"download)")
        _drop_transport(tv4z_path)   # only ever AFTER the sha gate above
        return local
    except Exception as exc:
        try:
            if binpath.is_file():
                binpath.unlink()
        except OSError:
            pass
        # 0.1.11: a decode that died because the disk is FULL must not fall
        # back to the raw bin — that is a LARGER file on the same full
        # volume, doomed by construction after a multi-GB wasted download.
        # Two detectors, because the C decoder's own ENOSPC says only
        # "tv4z: write failed" (no errno text): the exception itself, and —
        # after the partial bin above is deleted — whether the volume could
        # even hold the raw bin the fallback would fetch. Every OTHER decode
        # failure keeps the fallback.
        if _is_enospc(exc):
            _exit_disk_full(f"decoding {tv4z_name} in the model cache")
        if bin_bytes:
            try:
                free = shutil.disk_usage(local).free
            except OSError:
                free = None
            if free is not None and free < bin_bytes + _SPACE_MARGIN:
                root = _download_cache_root()
                print(f"[{BRAND}] ERROR: the transport decode failed ({exc}) "
                      f"and the raw-bin fallback cannot fit either:\n"
                      f"[{BRAND}]   model cache dir : {root}\n"
                      f"[{BRAND}]   fallback needs  : ~{_gb(bin_bytes)} "
                      f"({bin_name})\n"
                      f"[{BRAND}]   space free      : {_gb(free)}\n"
                      f"[{BRAND}] Not starting a {_gb(bin_bytes)} download "
                      f"onto a volume that cannot hold it. Free up\n"
                      f"[{BRAND}] space there, or move the cache to a disk "
                      f"with room and re-run:\n"
                      + _space_fix_lines(),
                      file=sys.stderr, flush=True)
                raise SystemExit(1) from None
        _note(f"transport decode unavailable ({exc}); downloading the raw "
              f"{bin_name} instead (same result, larger download)")
        return _snapshot(f"downloading the raw {bin_name}", repo_id=spec,
                         allow_patterns=HF_RUNTIME_PATTERNS)


def _config_dir_for(container: Path) -> Path:
    import fermion  # registers trtc_v4
    sha = hashlib.sha256(str(container.resolve()).encode()).hexdigest()[:16]
    cfg = _cache_dir() / f"cfg-{sha}"
    fermion.write_transformers_config(container, cfg)
    return cfg


def container_for(kind: str, path: Path) -> Path:
    """The .bin the runtime opens, whichever way the model was specified."""
    if kind == "trtc-dir":
        from fermion._container import find_container
        return find_container(path)
    return Path(path)


_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
                    "added_tokens.json", "tokenizer.model", "chat_template.jinja", "chat_template.json")


def _tokenizer_files_dir(src) -> str | Path:
    """A directory holding ONLY the tokenizer files of `src` (a Hub snapshot or a config dir).

    AutoTokenizer.from_pretrained(dir) first parses the directory's config.json through the BASE PretrainedConfig to learn
    the tokenizer class; transformers >= 5 then does getattr(torch, config["dtype"]) and the published containers carry
    "dtype": "auto" — every bare `fermion serve` on the native runtime died there (0.2.0/0.2.1). The tokenizer needs none of
    that file: it is loaded from a sibling directory with the tokenizer files alone (symlinked, copied if symlinks are not
    allowed), so no model config is ever parsed for it. Anything unexpected falls back to `src` unchanged."""
    try:
        src = Path(src)
        if not src.is_dir() or not (src / "tokenizer_config.json").exists():
            return src
        if not (src / "config.json").exists():
            return src                       # nothing to shield against
        key = hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:16]
        out = _cache_dir() / f"tok-{key}"
        out.mkdir(parents=True, exist_ok=True)
        for name in _TOKENIZER_FILES:
            f = src / name
            if not f.exists():
                continue
            dst = out / name
            if dst.is_symlink() or dst.exists():
                if dst.is_symlink() and os.readlink(dst) == str(f.resolve()):
                    continue
                dst.unlink()
            try:
                dst.symlink_to(f.resolve())
            except OSError:
                shutil.copyfile(f, dst)
        return out
    except Exception:
        return src


def _tokenizer_src(kind: str, path: Path, *, write_config: bool) -> str | Path:
    """Where AutoTokenizer should load from (same rule on both backends)."""
    cfg_dir = path if kind == "trtc-dir" else (
        _config_dir_for(path) if write_config else None)
    if cfg_dir is not None and (Path(cfg_dir) / "tokenizer_config.json").exists():
        return _tokenizer_files_dir(cfg_dir)
    return TOKENIZER_FALLBACK


def load_trtc(kind: str, path: Path, device: str, dtype: str = "auto", *,
              backend: str = "auto", draft: bool = False,
              native_bin: str | None = None, kv_dtype: str = "fp32",
              yarn_factor: float | None = None,
              yarn_orig_max: int | None = None):
    """Load a TRTC v4 container on the fastest backend that can serve it.

    Returns `(model, tokenizer)` where `model` is either a torch
    `TrtcV4ForCausalLM` or a `fermion.native.NativeModel`. Both are accepted
    by `iter_generate`, which is the only place the difference is handled.
    """
    from fermion import native as _native
    from transformers import AutoTokenizer

    container = container_for(kind, path)
    nat = _native.try_load(
        container, device=device, backend=backend, draft=draft,
        explicit_bin=native_bin, kv_dtype=kv_dtype,
        yarn_factor=yarn_factor, yarn_orig_max=yarn_orig_max,
        on_note=lambda m: print(f"[{BRAND}] native runtime not used: {m}; "
                                f"falling back to the torch reference path "
                                f"(correct, but much slower).",
                                file=sys.stderr))
    if nat is not None:
        # No torch model is constructed at all on this path: the container is
        # mmapped by the C runtime, which is also why RSS stays flat.
        tok_src = _tokenizer_src(kind, path, write_config=False)
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        print(f"[{BRAND}] native runtime: {nat.binary.name} "
              f"({nat.threads} threads, {nat.integrity}) · "
              f"tokenizer={tok_src}", file=sys.stderr)
        return nat, tokenizer

    import fermion
    fermion.language_model_loader()   # registers trtc_v4 (lazy since 0.1.25)
    import torch
    from transformers import AutoModelForCausalLM

    # CLI default is bfloat16: fp16 (the loader's arch-3 default) NaN-overflows
    # inside 8B-scale MLP/attention internals (residual absmax ~15k measured
    # at 8B scale; NaN from layer 7 in fp16, clean in bf16). --dtype float16
    # reproduces the
    # fp16-identity receipts on containers gated that way (small twins).
    torch_dtype = torch.bfloat16 if dtype == "auto" else getattr(torch, dtype)
    cfg_dir = path if kind == "trtc-dir" else _config_dir_for(path)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(cfg_dir, dtype=torch_dtype)
    if device != "cpu":
        model = model.to(device)
    model.backend = "torch"
    load_s = time.time() - t0
    tok_src = cfg_dir if (Path(cfg_dir) / "tokenizer_config.json").exists() \
        else TOKENIZER_FALLBACK
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    print(f"[{BRAND}] loaded {path.name} in {load_s:.1f}s "
          f"(device={device}, dtype={torch_dtype}, tokenizer={tok_src})",
          file=sys.stderr)
    return model, tokenizer


# --------------------------------------------------------------- trtc paths
def _rep_penalty_processor(penalty: float, window: int):
    """Windowed CTRL-style repetition penalty, matching the native runtime.

    The C sampler penalises the unique ids in the last `pen-window` context
    tokens (prompt tail + generated) with `x>0 ? x/pen : x*pen`. gather/scatter
    over a window with duplicate ids writes the same penalised value for every
    duplicate, so this is the same rule, applied once per unique id.
    """
    import torch
    from transformers import LogitsProcessor

    class _WindowedRepetitionPenalty(LogitsProcessor):
        def __init__(self, penalty: float, window: int):
            self.penalty = float(penalty)
            self.window = int(window)

        def __call__(self, input_ids, scores):
            ids = input_ids[:, -self.window:] if self.window > 0 else input_ids
            picked = torch.gather(scores, 1, ids)
            picked = torch.where(picked < 0, picked * self.penalty,
                                 picked / self.penalty)
            return scores.scatter(1, ids, picked)

    return _WindowedRepetitionPenalty(penalty, window)


def _finish(result: dict | None, ids, gen_ids, wall: float, max_new: int,
            backend: str) -> None:
    """Common tail for both backends: usage stats + the identity-harness hook."""
    if result is not None:
        result.update(prompt_tokens=len(ids), completion_tokens=len(gen_ids),
                      generated_ids=gen_ids, wall=wall, backend=backend,
                      finish_reason="length" if len(gen_ids) >= max_new
                      else "stop")
    dump = os.environ.get("FERMION_DUMP_IDS")   # token-identity harness hook
    if dump:
        Path(dump).write_text(json.dumps(gen_ids))


def _native_generate(model, tokenizer, ids, *, max_new: int, temperature: float,
                     top_p: float, rep_penalty: float, pen_window: int,
                     stop, ignore_eos: bool, result: dict | None):
    """Native half of `iter_generate`. Same yields, same `result` contract.

    Streams `TOK <id>` lines out of the C runtime and detokenises
    incrementally. The decode is redone over the whole id list each step
    (cheap at these lengths) rather than per token, because per-token decode
    splits multi-byte characters; a partial sequence shows up as a trailing
    U+FFFD and we simply hold the piece back until the next token completes it.
    """
    from fermion import native as _native

    stop_ids = [] if ignore_eos else _native.eos_ids(tokenizer)
    gen: list[int] = []
    shown = ""
    flag = {"stop": False}
    t0 = time.time()
    try:
        for tid in model.stream_ids(ids, max_new=max_new,
                                    temperature=temperature, top_p=top_p,
                                    rep_penalty=rep_penalty,
                                    pen_window=pen_window, stop_ids=stop_ids,
                                    should_stop=lambda: flag["stop"]):
            gen.append(tid)
            text = tokenizer.decode(gen, skip_special_tokens=True)
            if text.endswith("�"):      # mid-character; wait for the rest
                continue
            if text != shown:
                yield text[len(shown):] if text.startswith(shown) else text
                shown = text
            if stop and any(s in shown for s in stop):
                flag["stop"] = True
    except _native.NativeUnavailable:
        # Fill in usage for whatever DID come out before re-raising, so the
        # caller's rescue (iter_generate) can close a partial turn with honest
        # numbers instead of zeros. Whether to fall back or salvage is its call.
        if gen:
            _finish(result, ids, gen, time.time() - t0, max_new, "native")
        raise
    _finish(result, ids, gen, time.time() - t0, max_new, "native")
    if result is not None:
        # The runtime's own steady-state decode rate, which is the quantity
        # every published tok/s figure is. It is NOT the wall-clock rate the
        # caller computes: each call also pays one process start plus the
        # container map/repack, which the wall number includes and this one
        # does not. Both get printed; neither is allowed to hide the other.
        result["decode_tok_s"] = getattr(model, "last_tok_s", None)


def _torch_twin(nat, device: str):
    """Build the torch reference model for a NativeModel's own container.

    Only used by `iter_generate`'s NativeUnavailable rescue: at that point we
    hold a `NativeModel`, which has no forward pass, so the fallback has to
    construct the twin from the same .bin. dtype is the CLI default (bfloat16,
    the range-safe choice at 8B scale) because the failing path never carried
    a dtype down here.
    """
    import torch
    import fermion  # noqa: F401  (registers trtc_v4)
    from transformers import AutoModelForCausalLM

    container = Path(nat.container)
    model = AutoModelForCausalLM.from_pretrained(_config_dir_for(container),
                                                 dtype=torch.bfloat16)
    if device != "cpu":
        model = model.to(device)
    model.backend = "torch"
    return model.eval()


def iter_generate(model, tokenizer, ids, *, max_new: int, temperature: float,
                  device: str, top_p: float = 1.0,
                  rep_penalty: float = CLI_REP_PENALTY,
                  pen_window: int = SHIP_PEN_WINDOW,
                  stop: list[str] | None = None, ignore_eos: bool = False,
                  assistant_model=None, result: dict | None = None):
    """THE decode path. Yields decoded text pieces as they are produced.

    Dispatches on the loaded model's backend; both halves honour the same
    sampler contract and fill `result` the same way.

    Sampler mapping, torch side (see SHIP_* above):
      temperature == 0 -> deterministic argmax (`do_sample=False`) with the
        repetition penalty STILL APPLIED — transformers runs logits processors
        on the greedy path too, so unlike the C binary we never silently drop
        a rep-pen setting at temp 0. This is the "graded greedy" path.
      temperature  > 0 -> `do_sample=True` with temperature + top_p.
    The native side reproduces exactly that contract on a runtime that
    hard-disables its sampler at temperature 0; how, and at what cost, is
    documented in `fermion/native.py`.

    `ignore_eos=True` decodes the full `max_new` window regardless of EOS —
    the fixed-length window `fermion bench` times.

    On exhaustion, `result` (if given) is filled with prompt_tokens,
    completion_tokens, generated_ids, wall, backend and finish_reason.
    """
    if getattr(model, "backend", "torch") == "native":
        if assistant_model is not None:      # cannot happen via the CLI
            raise SystemExit("speculative decoding needs --backend torch")
        # `native.stream_ids` can raise NativeUnavailable AFTER the caller has
        # started iterating: the binary is exec'd, and the exit-code check runs
        # in the generator's `finally`. Until 0.1.7 that surfaced as a raw
        # traceback out of `chat`/`serve` instead of the torch fallback every
        # other native failure mode gets (open since 0.1.4). Catch it here:
        #  * nothing emitted yet -> warn and fall through to the torch path,
        #    which is what try_load()'s on_note path would have done had the
        #    failure been visible at load time;
        #  * tokens already emitted -> we cannot re-run without duplicating
        #    output, so close the turn cleanly with what we have and warn.
        from fermion.native import NativeUnavailable
        emitted = 0
        try:
            for piece in _native_generate(
                    model, tokenizer, ids, max_new=max_new,
                    temperature=temperature, top_p=top_p,
                    rep_penalty=rep_penalty, pen_window=pen_window,
                    stop=stop, ignore_eos=ignore_eos, result=result):
                emitted += 1
                yield piece
            return
        except NativeUnavailable as exc:
            if emitted:
                print(f"[{BRAND}] WARNING: the native runtime failed after "
                      f"{emitted} pieces ({exc}); returning the partial "
                      f"completion. Re-run with --backend torch for the "
                      f"reference path.", file=sys.stderr)
                if result is not None:
                    result.setdefault("finish_reason", "stop")
                    result["native_error"] = str(exc)
                return
            print(f"[{BRAND}] WARNING: the native runtime failed mid-stream "
                  f"({exc}); falling back to the torch reference path "
                  f"(correct, but much slower).", file=sys.stderr)
            model = _torch_twin(model, device)
            if result is not None:
                result["native_error"] = str(exc)

    import torch
    from threading import Thread
    from transformers import LogitsProcessorList, TextIteratorStreamer

    input_ids = torch.tensor([ids], device=device)
    kwargs = dict(input_ids=input_ids, max_new_tokens=max_new,
                  eos_token_id=tokenizer.eos_token_id,
                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if ignore_eos:
        kwargs["min_new_tokens"] = max_new
    # THE CLI OWNS THE SAMPLER. Every knob the model repo's
    # generation_config.json could also set is pinned here, because HF merges
    # generation_config into generate() and anything we leave unset silently
    # comes from that file:
    #   * repetition_penalty -> HF builds its OWN full-context
    #     RepetitionPenaltyLogitsProcessor. It is a DIFFERENT CLASS from the
    #     windowed one below, so _merge_criteria_processor_list does not
    #     dedupe them: both run, and the effective penalty inside the window
    #     is rep_penalty**2 (measured 1.1025 for a 1.05 setting). Pinning it
    #     to 1.0 stops HF adding its processor at all, so exactly one applies.
    #   * top_k -> GenerationConfig's default is 50, so the sampled path had an
    #     undocumented top-k we never measured. 0 disables it; top-p is the
    #     only shortlist knob this CLI exposes.
    kwargs["repetition_penalty"] = 1.0
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p,
                      top_k=0)
    else:
        kwargs.update(do_sample=False)
    if rep_penalty and float(rep_penalty) != 1.0:
        kwargs["logits_processor"] = LogitsProcessorList(
            [_rep_penalty_processor(rep_penalty, pen_window)])
    if stop:
        kwargs["stop_strings"] = list(stop)
        kwargs["tokenizer"] = tokenizer
    if assistant_model is not None:
        kwargs["assistant_model"] = assistant_model
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                    skip_special_tokens=True)
    kwargs["streamer"] = streamer

    box: dict = {}

    def _worker():
        try:
            box["out"] = model.generate(**kwargs)
        except BaseException as exc:          # noqa: BLE001 - re-raised below
            box["err"] = exc
            try:                              # never leave the reader blocked
                streamer.end()
            except Exception:                 # pragma: no cover
                pass

    t0 = time.time()
    thread = Thread(target=_worker, daemon=True)
    thread.start()
    try:
        for piece in streamer:
            yield piece
    finally:
        thread.join()
    if "err" in box:
        raise box["err"]

    wall = time.time() - t0
    out = box.get("out")
    gen_ids = out[0][len(ids):].tolist() if out is not None else []
    _finish(result, ids, gen_ids, wall, max_new, "torch")


def _generate(model, tokenizer, ids, max_new: int, temperature: float,
              device: str, stream: bool = True,
              rep_penalty: float = CLI_REP_PENALTY,
              pen_window: int = SHIP_PEN_WINDOW,
              top_p: float = SHIP_TOP_P) -> tuple[str, int, float]:
    t0 = time.time()
    pieces = []
    res: dict = {}
    for piece in iter_generate(model, tokenizer, ids, max_new=max_new,
                               temperature=temperature, device=device,
                               top_p=top_p, rep_penalty=rep_penalty,
                               pen_window=pen_window, result=res):
        pieces.append(piece)
        if stream:
            print(piece, end="", flush=True)
    wall = time.time() - t0
    text = "".join(pieces)
    # the tokens actually emitted, not a re-encode of the decoded text: those
    # two differ whenever detokenisation is not a bijection, and the rate we
    # print is a claim about the runtime, so it has to count runtime tokens.
    n_tok = res.get("completion_tokens", len(tokenizer.encode(text)) if text
                    else 0)
    if stream:
        print()
    decode = res.get("decode_tok_s")
    extra = f" · {decode:.1f} tok/s decode" if decode else ""
    print(f"[{BRAND}] {n_tok} tokens in {wall:.1f}s = "
          f"{n_tok / wall if wall else 0:.2f} tok/s wall{extra} "
          f"[{res.get('backend', 'torch')}]", file=sys.stderr)
    return text, n_tok, wall


def _chat_ids(tokenizer, messages, tools=None) -> list[int]:
    """Chat-template the messages. `tools`, when given, goes to the template.

    The shipped `tokenizer_config.json` chat template already has a `tools`
    branch: it renders the `<tools>…</tools>` block, keeps the default
    "You are Neutrino-1…" identity line, and knows how to replay assistant
    `tool_calls` and `role: "tool"` turns. We pass `tools=` straight through
    rather than hand-building a preamble, so the prompt we serve is the one the
    model's own repo declares.

    `tools` is added to the call ONLY when non-empty: `tools=None` is
    documented as equivalent to omitting it, but the no-tools prompt is
    covered by a byte-identity gate against 0.1.6, so we do not rely on that.
    """
    extra = {"tools": list(tools)} if tools else {}
    try:
        out = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            **extra)
    except TypeError:  # tokenizers without the enable_thinking kwarg
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            **extra)
    # transformers 4.x returned list[int]; 5.x fast tokenizers return a
    # tokenizers.Encoding (or BatchEncoding); torch.tensor accepts neither.
    if hasattr(out, "input_ids"):
        out = out.input_ids
    out = getattr(out, "ids", out)
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def chat_trtc(model, tokenizer, device: str, max_new: int, temperature: float,
              rep_penalty: float = SHIP_REP_PENALTY,
              pen_window: int = SHIP_PEN_WINDOW,
              top_p: float = SHIP_TOP_P):
    print(f"[{BRAND}] chat REPL. /exit to quit, /reset to clear history.")
    graded = (temperature == SHIP_TEMPERATURE
              and rep_penalty == SHIP_REP_PENALTY)
    which = "graded shipping config" if graded else "custom"
    print(f"[{BRAND}] sampling temp={temperature} top_p={top_p} "
          f"rep_pen={rep_penalty} window={pen_window} ({which}); "
          f"`fermion generate` is deterministic greedy by default.",
          file=sys.stderr)
    messages = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            messages = []
            continue
        messages.append({"role": "user", "content": user})
        ids = _chat_ids(tokenizer, messages)
        print("model> ", end="", flush=True)
        text, _, _ = _generate(model, tokenizer, ids, max_new, temperature,
                               device, rep_penalty=rep_penalty,
                               pen_window=pen_window, top_p=top_p)
        messages.append({"role": "assistant", "content": text})


# --------------------------------------------------------------- tiny paths
def tiny_reply(buf: bytes, prompt: str) -> tuple[str, int, int]:
    """(text, prompt_tokens, completion_tokens) from the float64 reference."""
    from fermion._vendor import tiny_ref
    if buf[:4] == b"QRB8":
        gen, text, vocab = tiny_ref.qrb8_greedy(buf, prompt)
        stoi = {w: i for i, w in enumerate(vocab)}
        n_prompt = len([stoi["A:"]] + prompt.split() + ["\n", "B:"])
    else:
        gen, text, vocab = tiny_ref.qrl8_greedy(buf, prompt)
        n_prompt = len(tiny_ref.BPETok(vocab).encode(f"A: {prompt}\nB:"))
    return text, n_prompt, len(gen)


def _tiny_reply(buf: bytes, prompt: str) -> str:
    return tiny_reply(buf, prompt)[0]


def _tiny_ignored_flags(args) -> list[str]:
    """The sampler flags this tiny run cannot honour, as the user typed them.

    The tiny QRB8/QRL8 path decodes on the float64 reference every banked
    receipt (python/node/chromium/webkit, 12/12 greedy) is gated against:
    single-turn, GREEDY, fixed token budget (30 QRB8 / 60 QRL8 tokens,
    hardcoded in _vendor/tiny_ref.py). None of the sampler knobs exist on
    that path, and before the 0.1.12 re-gate a `--temperature 0.8` on a tiny
    artifact was swallowed SILENTLY. Detection compares against THIS
    subcommand's own defaults (chat and generate deliberately differ; see
    _common_parser), because argparse does not record which flags were
    actually typed: a flag left at its default is indistinguishable from an
    untyped one — and also changes nothing, so it stays quiet.
    """
    if args.cmd == "chat":
        ref = _common_parser(temperature=SHIP_TEMPERATURE,
                             rep_penalty=SHIP_REP_PENALTY,
                             max_new=CHAT_MAX_NEW)
    else:                                   # generate
        ref = _common_parser(temperature=GENERATE_TEMPERATURE,
                             rep_penalty=CLI_REP_PENALTY)
    defaults = ref.parse_args([])
    moved = [f"--{name.replace('_', '-')} {getattr(args, name)}"
             for name in ("temperature", "top_p", "rep_penalty", "pen_window",
                          "max_new")
             if getattr(args, name) != getattr(defaults, name)]
    if getattr(args, "raw", False):
        moved.append("--raw")
    return moved


def _warn_tiny_ignored(args) -> None:
    """stderr WARN, nothing else: the tiny decode itself is deliberately
    untouched — greedy bit-exactness (12/12 prompts x 4 engines) is a ship
    gate, so this function must never influence what gets generated."""
    moved = _tiny_ignored_flags(args)
    if moved:
        _note("WARNING: tiny QRB8/QRL8 artifacts decode on the fixed float64 "
              "reference path (single-turn, greedy, fixed token budget) — "
              "ignoring: " + ", ".join(moved) + ". Sampled decode exists only "
              "in the browser demo; CLI tiny output stays byte-exact to the "
              "banked greedy receipts.")


def chat_tiny(path: Path):
    buf = path.read_bytes()
    kind = "QR (2.5 KB GRU)" if buf[:4] == b"QRB8" else "GIF (170 KB transformer)"
    print(f"[{BRAND}] tiny model chat: {path.name} [{kind}]. /exit to quit. "
          f"Single-turn, greedy, float64 reference decoder.")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        t0 = time.time()
        text = _tiny_reply(buf, user)
        print(f"model> {text}")
        print(f"[{BRAND}] {time.time() - t0:.2f}s", file=sys.stderr)


# --------------------------------------------------------------------- info
def _info(args, kind: str, path: Path) -> int:
    """`fermion info` — the "did my download work?" command, so it has to be
    able to answer NO.

    Until 0.1.3 this printed a header and exited 0 for any file whose first
    60 bytes were well formed, which meant an interrupted download — the
    single most likely thing to go wrong with a 3.9 GB fetch — produced a
    confident green. It now proves the whole file is present (and matches
    the published sha256 when the model repo's MANIFEST.json is on disk)
    before it says anything reassuring, and exits 1 when it cannot.
    """
    from fermion._container import ContainerError, check_container, find_container

    if kind == "tiny":
        # 2.5 KB - 170 KB artifacts: size + magic, still without slurping.
        size = path.stat().st_size
        print(f"{path}: {read_magic(path).decode()} artifact, {size} bytes")
        return 0

    import fermion
    container = path if kind == "trtc-bin" else find_container(path)
    try:
        h = fermion.read_header(container)          # 60 bytes
        report = check_container(
            container, checksum=not args.no_checksum,
            on_note=lambda m: print(f"[{BRAND}] {m}", file=sys.stderr))
    except (ContainerError, ValueError) as exc:
        print(f"{container}: INTEGRITY CHECK FAILED\n  {exc}", file=sys.stderr)
        print(f"[{BRAND}] this file is not a usable model. Delete it and "
              f"fetch it again.", file=sys.stderr)
        return 1

    print(f"{container}: TRTC v{h.version} arch={h.arch} "
          f"layers={h.n_layers} hidden={h.hidden} vocab={h.vocab} "
          f"({report['bytes'] / 2**30:.2f} GiB)")
    checks = {"structure": "length matches its own record structure",
              "manifest-bytes": "length matches MANIFEST.json",
              "manifest-sha256": "sha256 matches MANIFEST.json"}
    print(f"[{BRAND}] integrity OK: {report['bytes']:,} bytes · "
          + " · ".join(checks[c] for c in report["checked"]))
    if report.get("sha256"):
        print(f"[{BRAND}] sha256 {report['sha256']}")
    elif report["pin"] and args.no_checksum:
        print(f"[{BRAND}] sha256 not verified (--no-checksum)")
    _print_backend(args, container)
    return 0


def _print_backend(args, container: Path) -> None:
    """Say, out loud, which decode path this machine will actually use.

    `fermion info` is the "did my install work?" command, and after 0.1.3 the
    single most consequential thing it can tell a user is whether they are
    about to get the 100x path or the reference path — the published tok/s
    numbers only describe the former.
    """
    from fermion import native as _native

    cov = _native.coverage()
    want = getattr(args, "backend", "auto")
    binary, why = _native.find_binary(
        container, explicit=getattr(args, "native_bin", None))
    if want == "torch":
        active, detail = "torch (reference)", "--backend torch requested"
    elif binary is None:
        active, detail = "torch (reference)", why
    elif getattr(args, "device", "cpu") != "cpu":
        active, detail = ("torch (reference)",
                          f"--device {args.device}: native is a CPU runtime")
    else:
        try:
            nat = _native.NativeModel(binary, container)
            active = f"native ({nat.binary.name})"
            detail = f"{nat.threads} threads · {nat.integrity}"
        except _native.NativeUnavailable as exc:
            active, detail = "torch (reference)", str(exc)
    print(f"[{BRAND}] decode backend: {active}")
    print(f"[{BRAND}]   {detail}")
    print(f"[{BRAND}]   platform {cov['this_platform']} · native runtimes "
          f"published for: {', '.join(cov['published_binaries'])}")
    if not active.startswith("native"):
        print(f"[{BRAND}]   the published tokens/s figures are native-runtime "
              f"figures; the torch path is correct but far slower.")


# --------------------------------------------------------------------- main
def _common_parser(*, temperature: float = 0.0,
                   rep_penalty: float = CLI_REP_PENALTY,
                   max_new: int = GENERATE_MAX_NEW
                   ) -> argparse.ArgumentParser:
    """A FRESH parent parser per subcommand, with its own sampler defaults.

    Do not hoist this back into a single shared `common` object, and do not
    reach for `set_defaults()` to give one subcommand a different default.
    `parser.set_defaults()` MUTATES `action.default` on the parent's Action
    objects, and `parents=[common]` shares those objects with every other
    subparser — so one subcommand's set_defaults silently rewrites the
    defaults of all the others, and the help text still prints the old value.
    That is exactly how `verify`'s `set_defaults(max_new=32)` and an earlier
    `serve` `set_defaults(temperature=0.01, rep_penalty=1.05)` leaked into
    `chat`/`generate`, turning the documented greedy default into a sampled
    one (caught 2026-07-27). Pass the differing defaults in here instead:
    each subcommand then owns its Action objects AND prints honest `--help`.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--allow-legacy-gguf", action="store_true",
                   help="run a GGUF published before Neutrino packs carried a "
                        "provenance marker")
    p.add_argument("--model", default=DEFAULT_MODEL_ID,
                   help="the same as the MODEL positional (the model comes first: "
                        "`fermion serve phonon-2`); a repo id, alias, local model "
                        "directory or TRTC .bin")
    p.add_argument("--device", default="cpu",
                   help="cpu (default; the gated reference path) or mps "
                        "(experimental)")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"],
                   help="activation dtype for TRTC containers "
                        "(auto = bfloat16, range-safe at 8B scale)")
    # 0.1.7: `chat` passes 1024. 256 clipped three otherwise-CORRECT answers
    # mid-conclusion during the behavioural deep-dive — a chat REPL that stops
    # talking halfway through its own reasoning reads as a model failure, not a
    # budget. `generate` keeps 256: it is the scripted surface, where a small
    # fixed budget is the point.
    p.add_argument("--max-new", type=int, default=max_new,
                   help=f"max new tokens (default {max_new})")
    p.add_argument("--temperature", type=float, default=temperature,
                   help=f"sampling temperature (default {temperature}). "
                        + SPLIT_NOTE
                        + (" 0 = greedy." if temperature == 0.0 else
                           " Pass 0 for deterministic greedy."))
    p.add_argument("--top-p", type=float, default=SHIP_TOP_P,
                   help=f"nucleus top-p, only used when temperature > 0 "
                        f"(default {SHIP_TOP_P})")
    p.add_argument("--rep-penalty", type=float, default=rep_penalty,
                   help=f"CTRL-style repetition penalty over the last "
                        f"--pen-window tokens (default {rep_penalty}"
                        + (" = off" if rep_penalty == 1.0 else "")
                        + f"; the graded shipping config is "
                          f"{SHIP_REP_PENALTY}). Applies on the greedy path "
                          f"too, unlike the C runtime, which no-ops every "
                          f"sampler flag at temperature 0.")
    p.add_argument("--pen-window", type=int, default=SHIP_PEN_WINDOW,
                   help=f"repetition-penalty window "
                        f"(default {SHIP_PEN_WINDOW})")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "native", "torch"],
                   help="decode backend. auto (default) = the prebuilt "
                        "bin/fermion-run-* runtime from the model repo when "
                        "one exists for this platform, else torch. native = "
                        "require it (error out rather than silently run "
                        "slow). torch = the reference path.")
    p.add_argument("--native-bin", default=None,
                   help="explicit path to a fermion-run binary (overrides "
                        "discovery; same as $FERMION_NATIVE_BIN)")
    p.add_argument("--kv-dtype", default="fp16",
                   choices=["fp16", "fp32", "int8"],
                   help="KV-cache storage dtype (native runtime, 0.1.10). "
                        "fp16 (DEFAULT): HALVES KV memory (288 -> 144 "
                        "KiB/token on the 8B), fp32 compute over fp16 storage "
                        "-- near-tie tokens can differ from fp32 but quality "
                        "is flat (see model card). fp32: the exactness escape "
                        "hatch, byte-identical to 0.1.9. int8: QUARTERS KV "
                        "memory (per-vector scale), opt-in for long context; "
                        "quality is coarser -- see the card")
    p.add_argument("--yarn-factor", type=float, default=None,
                   help="EXPERIMENTAL (native runtime, 0.1.10): enable YaRN "
                        "rope scaling with this factor, extending the "
                        "addressable context toward factor x 40,960 "
                        "positions. The rope math is the "
                        "transformers-validated YaRN twin. LEAVE THIS OFF "
                        "unless you actually need more than 40,960 tokens: "
                        "YaRN also applies a global attention-temperature "
                        "term (0.1*ln(factor)+1) that is NOT gated by "
                        "position, so it changes output at EVERY length, "
                        "short prompts included. It extends addressable "
                        "positions, it does not improve recall -- our own "
                        "needle probe measures retrieval falling off well "
                        "inside the native window (see the model card). KV "
                        "memory grows linearly with context: pair with "
                        "--kv-dtype int8. Default off (standard rope)")
    p.add_argument("--yarn-orig-max", type=int, default=None,
                   help="YaRN original_max_position_embeddings (default: the "
                        "model's max_pos, 40,960; Qwen3's official recipe "
                        "uses 32,768)")
    return p


def _backend_kwargs(args) -> dict:
    # 0.1.14: `draft` carries the SPEC (path/repo string) so the native
    # backend can attach it to the runner's own --draft; falsy stays None.
    # `--draft auto` resolves to the canonical draft ONCE, here, so the
    # native path, the torch path and the CLI notes all see one real path.
    d = getattr(args, "draft", None) or None
    if d:
        from fermion.draft import resolve_spec
        d = resolve_spec(d)
        args.draft = d
    return {"backend": getattr(args, "backend", "auto"),
            "draft": d,
            "native_bin": getattr(args, "native_bin", None),
            "kv_dtype": getattr(args, "kv_dtype", "fp32"),
            "yarn_factor": getattr(args, "yarn_factor", None),
            "yarn_orig_max": getattr(args, "yarn_orig_max", None)}


def main() -> None:
    # Shield notes (a torchaudio that cannot load, ...) go to stderr the moment
    # a shield fires — which since 0.1.25 is when a model loads, not at
    # `import fermion`: `--version`, `--help` and `models` load nothing heavy.
    from fermion import _compat
    _compat.announce_notes()
    ap = argparse.ArgumentParser(
        prog=BRAND,
        description="The Phonon speech recognition engine and the Neutrino language "
                    "models from one command. Name the model: `fermion transcribe phonon-2 meeting.wav`.",
        epilog="disk space: model downloads land in the Hugging Face cache "
               "(~/.cache/huggingface/hub — your HOME volume, wherever your "
               "big disk is mounted). The 8B needs ~4 GB free there — ~6.5 GB "
               "on Apple Silicon during the first run only (transport + "
               "decoded container; the transport is deleted after the "
               "sha-verified decode, steady state ~3.9 GB; "
               "FERMION_KEEP_TRANSPORT=1 keeps it). The CLI checks before "
               "downloading. Small home volume? Put the cache on the big "
               "disk: FERMION_CACHE_DIR=/big/disk/fermion (this CLI's models "
               "only) or HF_HOME=/big/disk/hf (the whole HF cache).")
    ap.add_argument("--version", action="version", version=f"{BRAND} {VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # chat is a conversational surface -> graded shipping config (sampled).
    c = sub.add_parser("chat",
                       parents=[_common_parser(temperature=SHIP_TEMPERATURE,
                                               rep_penalty=SHIP_REP_PENALTY,
                                               max_new=CHAT_MAX_NEW)],
                       help="interactive REPL (sampled at the shipping config)")
    c.add_argument("model_pos", nargs="?", metavar="MODEL",
                   help="the language model: neutrino-8b, neutrino-0.6b-chat, a repo id or a local container; "
                        "`fermion chat neutrino-8b`. A bare `fermion chat` prints the model list and exits 2")
    c.set_defaults(model=None)
    for _a in c._actions:
        if "--model" in getattr(_a, "option_strings", ()):
            _a.help = argparse.SUPPRESS
    # generate is the scriptable surface -> deterministic greedy.
    g = sub.add_parser("generate",
                       parents=[_common_parser(
                           temperature=GENERATE_TEMPERATURE,
                           rep_penalty=CLI_REP_PENALTY)],
                       help="one-shot completion (deterministic greedy)")
    g.add_argument("model_pos", nargs="?", metavar="MODEL",
                   help="the language model: neutrino-8b, neutrino-0.6b, a repo id or a local container; "
                        "`fermion generate neutrino-8b 'prompt text'`. A bare `fermion generate` prints the model list and exits 2")
    g.add_argument("prompt", nargs="?", help="prompt text")
    g.set_defaults(model=None)
    for _a in g._actions:
        if "--model" in getattr(_a, "option_strings", ()):
            _a.help = argparse.SUPPRESS
    g.add_argument("--raw", action="store_true",
                   help="no chat template; feed the prompt verbatim")
    i = sub.add_parser("info", parents=[_common_parser()],
                       help="print model header/metadata and verify the file "
                            "on disk is complete and unmodified")
    i.add_argument("--no-checksum", action="store_true",
                   help="skip the sha256 comparison against the model repo's "
                        "MANIFEST.json (the length checks always run)")

    # serve ships the GRADED config; it gets its own parent so those defaults
    # cannot reach chat/generate (and so `serve --help` prints them honestly).
    # serve is a conversational surface like chat, so it shares chat's raised
    # default completion budget (the old 256 clipped correct answers in the
    # behavioural deep-dive: 15/81 truncated).
    s = sub.add_parser("serve",
                       parents=[_common_parser(temperature=SHIP_TEMPERATURE,
                                               rep_penalty=SHIP_REP_PENALTY,
                                               max_new=CHAT_MAX_NEW)],
                       help="OpenAI-compatible HTTP endpoint (localhost)")
    s.add_argument("model_pos", nargs="?", metavar="MODEL",
                   help="the model to serve: phonon-2 (speech endpoints), neutrino-8b (chat endpoints), any alias, repo id "
                        "or local model directory; `fermion serve phonon-2`. --model MODEL means the same thing; a bare "
                        "`fermion serve` prints the model list and exits 2 (nothing is downloaded)")
    s.set_defaults(model=None)
    s.add_argument("--host", default="127.0.0.1",
                   help="bind address (default 127.0.0.1 = localhost only; "
                        "the server is UNAUTHENTICATED unless --api-key)")
    s.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    s.add_argument("--unix-socket", default=None, metavar="PATH",
                   help="serve on a Unix-domain socket at PATH instead of "
                        "--host/--port (dictation tools and local agents: "
                        "one warm engine per machine, filesystem permissions "
                        "instead of --api-key; a stale socket file is replaced)")
    s.add_argument("--draft", default=None,
                   help="optional draft container for speculative decoding "
                        "(assisted generation; must share the tokenizer)")
    s.add_argument("--api-key", default=None,
                   help="require this bearer token on /v1/* requests")
    # SPEECH: which endpoints `serve` mounts is decided by the served MODEL,
    # not by a flag. `--model FermionResearch/Phonon-1` mounts POST
    # /v1/audio/transcriptions and nothing else; any other model keeps the chat
    # endpoints exactly as they were. There is deliberately NO --profile flag:
    # each speech model is its own repo, so the model already names the
    # profile (see fermion/transcribe.py's docstring).
    s.add_argument("--cors", action="store_true",
                   help="send Access-Control-Allow-Origin: * (browser clients)")
    s.add_argument("--threads", type=int, default=None, metavar="N",
                   help="CPU speech engine threads (same as FERMION_CPU_THREADS)")
    s.add_argument("--served-model-name", default=None,
                   help="id reported by /v1/models (default: container name)")
    s.add_argument("--max-new-ceiling", type=int, default=4096,
                   help="hard ceiling on a client's max_tokens (default 4096). "
                        "Requests above it are clamped, not rejected: a client "
                        "asking for 65k tokens on a CPU decode path is always "
                        "a mistake (some agent frameworks send 65536 on every step), "
                        "and until 0.1.7 we obliged")
    s.add_argument("--session-ctx", type=int, default=8192,
                   help="KV-cache capacity (positions) of the resident native "
                        "session runner (default 8192). KV RAM scales "
                        "linearly: ~288 KiB/position on the 8B (~2.3 GiB at "
                        "8192), or half that with --kv-dtype fp16. Requests "
                        "that need more fall back to one-shot mode for that "
                        "request")
    s.add_argument("--no-session", action="store_true",
                   help="disable the resident session runner (KV-prefix reuse "
                        "across requests) and spawn the native runtime per "
                        "request, as 0.1.6 did")

    # Stuck detector (fermion/_stuck.py; thresholds derived from an
    # OpenHands-style production pattern set). Serve-only:
    # `chat` renders no tools, so there is no loop for it to break.
    s.add_argument("--stuck-detector", default="enforce",
                   choices=["enforce", "observe", "off"],
                   help="agent-loop stuck detector (default enforce). The "
                        "request body carries the whole conversation, so a "
                        "loop going nowhere is visible server-side: an "
                        "identical tool call repeated, an A-B-A-B pair, or the "
                        "same tool returning the same error. enforce = on "
                        "trip, force ONE tools-disabled diagnosis turn and "
                        "then a final answer, so the client's loop ends. "
                        "observe = log + count only, output unchanged. off = "
                        "no analysis. Every trip prints a structured JSON line "
                        "to stderr; counters are on /health")
    s.add_argument("--stuck-repeat", type=int, default=3,
                   help="trip after this many argument-identical tool calls in "
                        "one conversation (default 3)")
    s.add_argument("--stuck-alt", type=int, default=6,
                   help="trip when the last N tool calls are a strict A-B-A-B "
                        "alternation between two fixed calls (default 6 = 3 "
                        "cycles; minimum 4)")
    s.add_argument("--stuck-error", type=int, default=2,
                   help="trip after this many CONSECUTIVE identical error "
                        "returns from the same tool (default 2)")
    s.add_argument("--stuck-whitelist", default="",
                   help="comma-separated tool names that are allowed to repeat "
                        "forever (poll/wait tools). Names whose own "
                        "name or description already implies polling are "
                        "detected automatically — this is for the ones that "
                        "do not")
    s.add_argument("--stuck-no-auto-whitelist", action="store_true",
                   help="turn OFF the automatic poll-type detection (name "
                        "tokens like poll/wait/status, or a description that "
                        "says it polls). Only do this if you know no tool of "
                        "yours is meant to be called repeatedly")

    # Sampling profile for TOOL-CARRYING requests only.
    s.add_argument("--tool-profile", default="graded",
                   choices=["graded", "vendor", "vendor-thinking", "antiloop"],
                   help="sampler defaults applied ONLY to requests that carry "
                        f"`tools` (default graded = the banked shipping config "
                        f"temp {SHIP_TEMPERATURE}/top-p {SHIP_TOP_P}/rep-pen "
                        f"{SHIP_REP_PENALTY}, i.e. NO CHANGE — every published "
                        f"number was measured there). vendor = Qwen3's own "
                        "non-thinking guidance (0.7/0.8) plus a stronger "
                        "repetition penalty, its anti-repetition advice for "
                        "quantized checkpoints; vendor-thinking = the "
                        "thinking-mode pair (0.6/0.95); antiloop = vendor with "
                        "the strongest penalty and a wider window. The "
                        "non-graded profiles are UNMEASURED on our containers "
                        "and are never a benchmark configuration. An explicit "
                        "temperature/top_p in the request always wins")

    # inspect/verify/bench + the chat/generate --draft flag: own modules, one
    # hook (fermion/_tools.py). Nothing above this line is touched.
    from fermion import _tools
    _tools.register(sub, _common_parser)   # factory: one fresh parent each

    # 0.2.4: the speech verbs take the model as a positional, so `fermion transcribe phonon-2 --json meeting.wav`
    # (an option between two positionals) must work; argparse stops the `words` list at the first option, so the
    # leftover non-option tokens are folded back into it here. Unknown OPTIONS are still errors.
    args, extra = ap.parse_known_args()
    if getattr(args, "cmd", None) in ("transcribe", "listen", "serve", "chat", "generate") and extra:
        stray = [t for t in extra if t.startswith("-")]
        if stray:
            ap.error(f"unrecognized arguments: {' '.join(stray)}")
        if args.cmd == "transcribe":
            args.words = list(getattr(args, "words", None) or []) + extra
        elif getattr(args, "model_pos", None) is None and len(extra) == 1:
            args.model_pos = extra[0]
        else:
            ap.error(f"unrecognized arguments: {' '.join(extra)}")
    elif extra:
        ap.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.cmd in ("chat", "generate"):
        from fermion import _modelarg
        _modelarg.take_model(args, args.cmd, f"fermion {args.cmd} neutrino-8b" + (" 'Write a haiku'" if args.cmd == "generate" else ""))
        if args.cmd == "generate" and not getattr(args, "prompt", None):
            ap.error("generate needs a prompt: `fermion generate neutrino-8b 'Write a haiku'`")

    if args.cmd == "serve":
        from fermion.server import serve
        serve(args)
        return

    # Commands that own their own model lifecycle run HERE, before
    # `resolve_model()` can download anything. `serve` above is the original
    # instance of this pattern; `_tools.EARLY` (models, transcribe) is the same
    # escape hatch made reusable. See the comment on `_tools.EARLY`.
    rc = _tools.dispatch_early(args)
    if rc is not None:
        raise SystemExit(rc)

    kind, path = resolve_model(args.model)

    rc = _tools.dispatch(args, kind, path)
    if rc is not None:
        raise SystemExit(rc)

    if args.cmd == "info":
        raise SystemExit(_info(args, kind, path))

    if kind == "tiny":
        _warn_tiny_ignored(args)      # stderr only; decode path unchanged
        if args.cmd == "chat":
            chat_tiny(path)
        else:
            buf = path.read_bytes()
            t0 = time.time()
            text = _tiny_reply(buf, args.prompt)
            print(text)
            print(f"[{BRAND}] {time.time() - t0:.2f}s", file=sys.stderr)
        return

    model, tokenizer = load_trtc(kind, path, args.device, args.dtype,
                                 **_backend_kwargs(args))
    from fermion.draft import enable as _enable_draft      # --draft, own module
    _enable_draft(model, args)
    if args.cmd == "chat":
        if getattr(model, "backend", "") == "native":
            # 0.1.7: one resident runner for the whole REPL. Kills the ~6 s
            # process start per turn AND the re-prefill of the entire history
            # (chat resends it every turn); each turn now prefills only its
            # own new tokens. Byte-identity with the one-shot path is gated.
            # FERMION_SESSION=0 restores 0.1.6
            # behaviour.
            model.enable_session()
            print(f"[{BRAND}] {model.session_note}", file=sys.stderr)
        chat_trtc(model, tokenizer, args.device, args.max_new, args.temperature,
                  rep_penalty=args.rep_penalty, pen_window=args.pen_window,
                  top_p=args.top_p)
    else:
        if args.raw:
            ids = tokenizer.encode(args.prompt)
        else:
            ids = _chat_ids(tokenizer, [{"role": "user", "content": args.prompt}])
        _generate(model, tokenizer, ids, args.max_new, args.temperature,
                  args.device, rep_penalty=args.rep_penalty,
                  pen_window=args.pen_window, top_p=args.top_p)


if __name__ == "__main__":
    main()
