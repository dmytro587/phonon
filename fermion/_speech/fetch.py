"""Download, cross-check, verify and unpack a Phonon profile.

THE DOWNLOAD PATH IS THE SAME ONE THE LLM USES, AND THAT MATTERS
----------------------------------------------------------------
`cli._download_hf()` fetches `HF_WEIGHTLESS_PATTERNS` FIRST — and the first
entry in that list is the bare pattern `"config.json"`, which fnmatch-matches
only the ROOT `config.json` of the repo. Speech goes through the same
`cli._snapshot()` wrapper with the same shape, so the first thing that crosses
the network for a speech model is also the repo's root `config.json`.

VERIFIED, because it is load-bearing for a reason that has nothing to do with
correctness: with `library_name: mlx` (which has no `countDownloads` override
in the Hub's `huggingface.js`), **the Hub counts requests to `config.json`
and does not count `resolve/main/*.tar.zst` transfers at all** — without that
fetch the repo would record zero downloads forever. The Mac app's downloader
does the same thing for the same reason.

So: every `fermion transcribe` that has to download registers the download it
is about to perform, exactly like the Mac app does.

THE CROSS-CHECK CONTRACT
------------------------
The client's own pinned SHA-256 stays authoritative. A repo's published
`config.json` is a cross-check, never an override: a client that finds a
disagreement must refuse the download and tell the user to update, not adopt
the value published there. `_cross_check()` implements exactly that —
a disagreement REFUSES before spending a multi-hundred-megabyte transfer; an
unreachable or unparseable config is noted and does not block (the pin we
already hold is the authority, so its absence costs a cross-check, not a
guarantee).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import os
import sys
from pathlib import Path

from fermion._brand import BRAND

# Small files only. Same shape and the same reason as
# `cli.HF_WEIGHTLESS_PATTERNS`: pull the metadata first, decide from it, and
# never let `snapshot_download` with no filter drag the whole repo (which for
# Phonon is 1.735 GB of archives across four profiles to run one).
#
# `config.json` FIRST, unanchored so it matches only the repo root — see the
# module docstring for why that single request is not optional.
SPEECH_WEIGHTLESS_PATTERNS = [
    "config.json",          # the cross-check AND the counted request
    "README.md",            # the model card
    "LICENSE",
    "NOTICE",
    "verify_install.py",    # the independent verifier the card points at
    "package_release_bps.py",   # the reference unpacker
]


class SpeechFetchError(Exception):
    pass


# ------------------------------------------------------------------ layout
def speech_root() -> Path:
    """Where unpacked speech profiles live.

    Under this CLI's own cache (`~/.cache/fermion`, honouring
    `$FERMION_CACHE_DIR` exactly as the LLM path does) rather than inside the
    Hub snapshot: `snapshot_download` materialises every file as a SYMLINK into
    a flat `blobs/` store, and the runtime opens a model DIRECTORY with a
    `config.json` and N safetensors shards next to each other. Unpacking into
    the snapshot would also mean re-unpacking on every revision bump.
    """
    from fermion.cli import _cache_dir, _cache_override
    override = _cache_override()
    root = Path(override) if override else _cache_dir()
    d = root / "speech"
    d.mkdir(parents=True, exist_ok=True)
    return d


def profile_dir(repo: str, unpack_dir: str) -> Path:
    """The directory the engine loads for one profile.

    Namespaced by REPO, because each speech profile now lives in its own repo
    and a user may well have two or three of them side by side. The last path
    element keeps the historical `unpack_dir` name (e.g. `model_v18_mlx_quint5`)
    so a directory produced here is interchangeable with one produced by the
    app's developer tree or by `package_release_bps.py unpack …` directly.
    """
    slug = repo.replace("/", "__")
    return speech_root() / slug / unpack_dir


def installed(repo: str, unpack_dir: str) -> bool:
    d = profile_dir(repo, unpack_dir)
    return is_model_dir(d)


def is_model_dir(d) -> bool:
    """Does this directory hold a runnable, fully-unpacked Phonon profile?

    Both files, not just one: `config.json` alone is what a HALF-unpacked tree
    looks like, and treating that as ready would hand the loader a directory
    with missing shards.
    """
    d = Path(d)
    return (d / "config.json").is_file() and (d / "packed_manifest.json").is_file()


# ------------------------------------------------------------- cross-check
def _cross_check(local: Path, repo: str, key: str, pin: dict) -> None:
    """Compare the repo's published facts against our pin. Refuse on conflict.

    Each speech repo now carries exactly ONE profile, described twice in its
    `config.json`: an `artifact` block (the canonical single-artifact form) and
    a one-entry `profiles.entries` list (the older shape). The `artifact` block
    is preferred and the list is the fallback, so this keeps working whichever
    of the two a future regeneration keeps.

    THE WRONG-REPO CHECK IS THE IMPORTANT ONE. With one repo per profile, the
    failure that actually costs a user a 581 MB download is asking a repo for a
    sibling's archive. `config.json` names both its own `model_id` and its own
    `artifact.profile`, so a mismatch on either is caught after one small
    request and before any archive is fetched.
    """
    from fermion.cli import _note
    cfg = local / "config.json"
    if not cfg.is_file():
        _note("the repo published no root config.json; using the pinned "
              "digest without a cross-check")
        return
    try:
        published = json.loads(cfg.read_text())
    except Exception as exc:                                # noqa: BLE001
        _note(f"could not parse the repo's config.json ({exc}); using the "
              f"pinned digest without a cross-check")
        return

    # Is this even the repo we think it is?
    model_id = published.get("model_id")
    if model_id and str(model_id).lower() != repo.lower():
        raise SpeechFetchError(
            f"{repo} publishes a config.json that identifies itself as "
            f"{model_id!r}. Refusing to download an archive from a repo that "
            f"does not agree about which model it is.")

    entry = published.get("artifact")
    if not isinstance(entry, dict):
        entries = {e.get("key"): e
                   for e in (published.get("profiles") or {}).get("entries") or []}
        entry = entries.get(key)
    if entry is None:
        _note(f"profile {key!r} is not described in {repo}'s config.json; "
              f"continuing against the pinned digest")
        return

    # Does this repo hold the profile we asked it for?
    got_profile = entry.get("profile")
    if got_profile and got_profile != key:
        raise SpeechFetchError(
            f"{repo} publishes the {got_profile!r} profile, but this client "
            f"asked it for {key!r}. Each Phonon model lives in its own repo; "
            f"refusing to fetch a profile from the wrong one.")

    for field in ("filename", "sha256", "download_bytes"):
        want, got = pin[field], entry.get(field)
        if got is not None and got != want:
            raise SpeechFetchError(
                f"the published {key} profile in {repo} disagrees with this "
                f"version of {BRAND}:\n"
                f"    {field}: published {got!r}\n"
                f"    {field}: pinned    {want!r}\n"
                f"Refusing the download rather than adopting the published "
                f"value. Upgrade with `pip install -U fermion-research`; if "
                f"that does not resolve it, the repo and this client are "
                f"describing different artifacts and neither should be "
                f"trusted until that is explained.")


# ------------------------------------------------------------------ zstd
def _zstd_reader(archive: Path):
    """A binary stream of the archive's DECOMPRESSED bytes.

    Three sources, in order of preference, because none of them is guaranteed:

      1. the `zstandard` PyPI package (an Apple-silicon wheel exists, and a
         speech user is installing wheels anyway);
      2. `compression.zstd`, stdlib from Python 3.14;
      3. the `zstd` command-line tool.

    The published quickstart and `package_release_bps.py:unpack` both shell out
    to `zstd`, which is what the release was verified with — but `zstd` is NOT
    installed by default on macOS (there is no `/usr/bin/zstd`; it arrives via
    Homebrew), so a CLI that only knew that route would fail on a clean Mac at
    the very last step of a 581 MB download. Every route feeds the identical
    byte-plane join and the identical per-member SHA-256 check below, so the
    artifact on disk is the same whichever one runs.
    """
    # RESOLVE THE SYMLINK FIRST. `snapshot_download` materialises every file in
    # `snapshots/<rev>/` as a SYMLINK into `blobs/<sha>`, and the `zstd`
    # command-line tool REFUSES to read a symlink: it prints
    # `is a symbolic link, ignoring` to stderr, writes NOTHING to stdout, and
    # does not fail loudly enough to be obvious. The result is an empty tar
    # stream and a baffling `ReadError: empty file` at the end of a
    # several-hundred-megabyte download, on every first run, for every user.
    # (`fermion/native.py:find_binary` documents the mirror image of this trap
    # for the native runner: there the fix is NOT to resolve, because resolving
    # walks out of the repo tree; here the fix IS to resolve, because we want
    # the blob itself and nothing near it.) The Python decoders below open the
    # path normally and follow symlinks by themselves, so resolving is
    # harmless for them and required for the CLI.
    archive = Path(archive).resolve()
    try:
        import zstandard                                  # type: ignore
        return zstandard.ZstdDecompressor().stream_reader(open(archive, "rb")), None
    except ImportError:
        pass
    try:
        from compression import zstd as _z                # type: ignore  # py3.14+
        return _z.ZstdFile(archive, "rb"), None
    except ImportError:
        pass
    exe = shutil.which("zstd")
    if exe:
        proc = subprocess.Popen([exe, "-q", "-d", "-c", str(archive)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        return proc.stdout, proc
    raise SpeechFetchError(
        "cannot decompress the model archive: none of the `zstandard` Python "
        "package, Python 3.14's `compression.zstd`, or the `zstd` command-line "
        "tool is available. Install either one:\n"
        "    pip install zstandard\n"
        "    brew install zstd")


# ----------------------------------------------------------------- unpack
def _unpack(archive: Path, dest: Path) -> int:
    """Byte-plane join + per-member SHA-256 verify, into `dest`.

    The transform and the verification are the vendored release packer's own
    (`fermion/_speech/_engine/package_release_bps.py`, byte-identical to the
    file published in the model repo): `join_file()` reassembles each tensor's
    byte planes and every member is checked against the manifest's
    `original_sha256` and `original_bytes` BEFORE it is written. A corrupt
    download fails at install time, not at inference time.

    Only the decompression source differs from the published `unpack()` — see
    `_zstd_reader`. Written to a temporary sibling and renamed at the end, so
    an interrupted unpack can never leave a half-populated directory that
    `installed()` would then call ready.
    """
    import tarfile

    from fermion._speech._engine import load as _engine_load
    bps = _engine_load("package_release_bps")

    staging = dest.with_name(dest.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    stream, proc = _zstd_reader(archive)
    manifest = None
    index: dict = {}
    written: list[str] = []
    try:
        try:
            tar = tarfile.open(fileobj=stream, mode="r|")
        except tarfile.ReadError as exc:
            # The decompressor produced nothing usable. Say which decompressor
            # and what it said, rather than letting tarfile's "empty file"
            # stand as the whole explanation.
            err = ""
            if proc is not None:
                try:
                    err = (proc.stderr.read() or b"").decode(
                        "utf-8", "replace").strip()
                except Exception:                           # pragma: no cover
                    pass
            raise SpeechFetchError(
                f"could not read {archive.name} as a zstd-compressed tar "
                f"({exc})." + (f" The decompressor said: {err}" if err else "")
            ) from exc
        with tar:
            for member in tar:
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                blob = handle.read()
                if member.name == "bps_manifest.json":
                    manifest = json.loads(blob)
                    index = {row["path"]: row for row in manifest["files"]}
                    continue
                if manifest is None:
                    raise SpeechFetchError(
                        "bps_manifest.json must be the first archive member")
                rel = (member.name[:-4] if member.name.endswith(".bps")
                       else member.name)
                row = index[rel]
                data = (bps.join_file(blob, row["transform"])
                        if "transform" in row else blob)
                if bps.sha256_bytes(data) != row["original_sha256"] \
                        or len(data) != row["original_bytes"]:
                    raise SpeechFetchError(
                        f"checksum mismatch unpacking {rel} — the archive is "
                        f"corrupt; delete it and re-run")
                target = staging / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                written.append(rel)
    finally:
        try:
            stream.close()
        except Exception:                                   # pragma: no cover
            pass
        if proc is not None and proc.wait() != 0:
            shutil.rmtree(staging, ignore_errors=True)
            raise SpeechFetchError("zstd decompression failed")
    if manifest is None:
        shutil.rmtree(staging, ignore_errors=True)
        raise SpeechFetchError("archive carried no bps_manifest.json")
    missing = {row["path"] for row in manifest["files"]} - set(written)
    if missing:
        shutil.rmtree(staging, ignore_errors=True)
        raise SpeechFetchError(f"archive is missing members: {sorted(missing)}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(dest)
    return len(written)


# ------------------------------------------------------------------ entry
def ensure(repo: str, key: str, pin: dict, *, quiet: bool = False) -> Path:
    """Make profile `key` of `repo` present on disk; return its directory.

    Idempotent and cheap when the profile is already unpacked: nothing is
    fetched, nothing is hashed, no network call is made.
    """
    from fermion.cli import (_gb, _hf_progress_bars_on, _note,
                             _preflight_space, _snapshot)
    from fermion._container import sha256_file

    if not repo:
        # `compact`: retained, never published, no repo exists. Refuse rather
        # than construct a URL that cannot resolve.
        raise SpeechFetchError(
            f"the {key!r} profile was cut before release and is not published "
            f"anywhere, so it cannot be downloaded. If you already have it "
            f"unpacked, point at the directory directly: "
            f"`{BRAND} transcribe --model /path/to/{pin['unpack_dir']} "
            f"audio.wav`. Otherwise use one of the published models "
            f"(`{BRAND} models`).")

    # 0.2.3: the audio-I/O probe runs HERE too, so no caller order can download a model onto a machine that cannot read
    # audio (round-3 verification called ensure() before the engine gate and pulled 164 MB it could never use). The CLI
    # entry points already gate before they fetch; this is the belt for library callers. FERMION_SPEECH_FETCH_ANYWAY=1
    # keeps a pure pre-download possible (moving a cache to another machine).
    if not os.environ.get("FERMION_SPEECH_FETCH_ANYWAY"):
        from fermion._speech import gate_cpu as _gate_cpu
        problem = _gate_cpu.audio_io_problem()
        if problem:
            raise SpeechFetchError(f"not downloading {repo} ({key}): {problem}")
    dest = profile_dir(repo, pin["unpack_dir"])
    if installed(repo, pin["unpack_dir"]):
        if not quiet:
            _note(f"using the cached {key} profile at {dest}")
        return dest

    # One process downloads and unpacks; any other `fermion` started at the
    # same time waits on this lock and then finds the finished model
    # (0.1.25; before, the second process unpacked into the same directory
    # and died on a half-written tree). Cross-process, cross-platform.
    from fermion._lock import CacheLock
    from fermion._brand import SPEECH_FAMILY
    with CacheLock(dest.with_name(dest.name + ".lock"),
                   what=f"the {SPEECH_FAMILY} model this command needs",
                   note=lambda m: print(m, file=sys.stderr, flush=True)) as lock:
        if installed(repo, pin["unpack_dir"]):
            _note(f"the {key} profile was unpacked by another fermion "
                  f"process meanwhile ({lock.waited:.0f} s wait); using it")
            return dest
        return _fetch_locked(repo, key, pin, dest)


def _fetch_locked(repo: str, key: str, pin: dict, dest: Path) -> Path:
    """Download, cross-check, verify and unpack — the caller holds the lock."""
    from fermion.cli import (_gb, _hf_progress_bars_on, _note,
                             _preflight_space, _snapshot)
    from fermion._container import sha256_file

    _note(f"fetching {repo} profile {key!r} from the Hugging Face Hub "
          f"({_gb(pin['download_bytes'])} download, "
          f"{_gb(pin['unpacked_bytes'])} unpacked; cached after)")
    _hf_progress_bars_on()

    # 1. metadata first — the cross-check, and the request the Hub counts.
    local = _snapshot("downloading the model metadata", repo_id=repo,
                      allow_patterns=SPEECH_WEIGHTLESS_PATTERNS)
    _cross_check(local, repo, key, pin)

    # 2. free space for the archive AND the tree it unpacks to. The archive is
    #    deleted afterwards, so peak is both and steady state is the tree.
    _preflight_space(local, repo,
                     pin["download_bytes"] + pin["unpacked_bytes"],
                     f"{_gb(pin['download_bytes'])} archive + "
                     f"{_gb(pin['unpacked_bytes'])} unpacked model")

    # 3. the archive itself, through the same snapshot path as everything else.
    local = _snapshot(f"downloading {pin['filename']}", repo_id=repo,
                      allow_patterns=SPEECH_WEIGHTLESS_PATTERNS
                      + [pin["filename"]])
    archive = local / pin["filename"]
    if not archive.is_file():
        raise SpeechFetchError(
            f"{repo} does not publish {pin['filename']}. If the repo is not "
            f"public yet this is expected; the model card and the four "
            f"archives go live together.")

    # 4. OUR pin, over the whole archive, before a single byte is unpacked.
    got = sha256_file(archive)
    if got != pin["sha256"]:
        raise SpeechFetchError(
            f"{pin['filename']} sha256 {got[:16]}… != the pinned "
            f"{pin['sha256'][:16]}… — the bytes are not the bytes we "
            f"published. Delete it from the Hugging Face cache and retry; if "
            f"it recurs, do not use this download.")
    _note(f"{pin['filename']} verified against the pinned sha256 "
          f"({got[:12]}…); unpacking…")

    n = _unpack(archive, dest)
    _note(f"unpacked {n} files to {dest} (every member SHA-256 verified)")
    return dest


def uninstall(repo: str, unpack_dir: str) -> bool:
    d = profile_dir(repo, unpack_dir)
    if d.is_dir():
        shutil.rmtree(d)
        return True
    return False
