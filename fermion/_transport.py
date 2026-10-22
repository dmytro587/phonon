"""Compressed download transport: pull the `.tv4z` and decode it to the `.bin`.

Every model repo carries two copies of the same weights: the runnable TRTC v4
`.bin` (3.88 GB for the 8B) and a lossless, entropy-coded `.tv4z` transport
(2.56 GB) that decodes back to that `.bin` BIT-IDENTICALLY. Until 0.1.9 the
loader pulled the raw `.bin`; this module lets the loader pull the smaller
`.tv4z` and decode it locally, which is what makes the advertised
"2.56 GB download" literally true for a fresh install.

The decoder is a prebuilt native binary, the exact sibling of the encoder
that produced the published transport — reusing it, rather than
re-implementing the entropy coder in Python, is both the only way to hit the
native tool's ~6 s decode (a pure-Python decode would run for tens of minutes)
AND the safest guarantee of the byte-identity gate the loader enforces
afterwards. Official PyPI wheels bundle the binary under `fermion/bin/`; a
source checkout carries no decoder, and every code path falls back to
downloading the raw `.bin` (same result, larger download).

CommonCrypto (macOS libSystem) supplies the decoder's MD5, so a runnable binary
exists only for Apple Silicon. Every other platform — and every failure — is
handled by the caller falling back to downloading the raw `.bin`, which is
exactly today's behaviour, so no user is ever left unable to run.
"""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

# (system, machine) -> the decoder basename bundled in fermion/bin/.
# Only Apple Silicon has a build (the decoder links CommonCrypto for MD5);
# anything absent here means "no local decode" -> the caller pulls the raw bin.
_DECODER_BINARIES = {
    ("Darwin", "arm64"): "tv4z-macos-arm64",
    ("Darwin", "aarch64"): "tv4z-macos-arm64",
}

_HASH_CHUNK = 1 << 24


class DecodeError(Exception):
    """The tv4z transport could not be decoded to a usable .bin."""


def _bin_dir() -> Path:
    return Path(__file__).resolve().parent / "bin"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def _sidecar_ok(binary: Path) -> tuple[bool, str]:
    """Verify `<binary>.sha256` before we execute the decoder."""
    pin = binary.with_name(binary.name + ".sha256")
    if not pin.is_file():
        return True, "no .sha256 sidecar (not verified)"
    try:
        want = pin.read_text().split()[0].strip().lower()
    except (OSError, IndexError):
        return True, "unreadable .sha256 sidecar (not verified)"
    got = _sha256(binary)
    if got != want:
        return False, f"decoder sha256 {got[:12]}… != pinned {want[:12]}…"
    return True, f"sha256 verified ({got[:12]}…)"


def _make_executable(binary: Path) -> None:
    """Restore the +x bit a wheel/hub client may have dropped (0644)."""
    try:
        mode = binary.stat().st_mode
    except OSError:
        return
    if not mode & 0o111:
        try:
            binary.chmod(mode | 0o755)
        except OSError:
            pass


def decoder_binary() -> Path | None:
    """A runnable tv4z decoder for this machine, or None.

    Order: $FERMION_TV4Z (explicit path) -> the wheel's fermion/bin/<tag> ->
    `tv4z` on $PATH. Returns None (raw-bin fallback) when nothing runnable and
    sha-verified is available for this platform.
    """
    explicit = os.environ.get("FERMION_TV4Z")
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None

    tag = _DECODER_BINARIES.get((platform.system(), platform.machine().lower()))
    if tag:
        cand = _bin_dir() / tag
        if cand.is_file():
            _make_executable(cand)
            ok, _ = _sidecar_ok(cand)
            if ok:
                return cand

    onpath = shutil.which("tv4z")
    return Path(onpath) if onpath else None


def can_decode() -> bool:
    return decoder_binary() is not None


def decode(tv4z_path: Path, out_bin: Path, *, threads: int | None = None,
           on_note=None) -> float:
    """Decode `tv4z_path` -> `out_bin` in place. Returns decode wall-seconds.

    Runs the C decoder with its own embedded-MD5 gate on (a first, cheap
    integrity check); the caller then re-verifies the sha256 of `out_bin`
    against the repo MANIFEST, which is the authoritative byte-identity gate.
    Raises `DecodeError` on any failure — the caller falls back to the raw bin.
    """
    binary = decoder_binary()
    if binary is None:
        raise DecodeError("no tv4z decoder available for this platform")
    ok, why = _sidecar_ok(binary)
    if not ok:
        raise DecodeError(f"refusing to run decoder: {why}")
    if threads is None:
        threads = max(1, min(os.cpu_count() or 8, 16))

    cmd = [str(binary), "d", str(tv4z_path), str(out_bin), str(int(threads))]
    if on_note:
        on_note(f"decoding {tv4z_path.name} -> {out_bin.name} "
                f"({threads} threads)…")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise DecodeError(f"could not launch decoder: {exc}") from exc
    secs = time.time() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise DecodeError(
            f"decoder exit {proc.returncode}: {' / '.join(tail) or 'no output'}")
    if not out_bin.is_file():
        raise DecodeError("decoder reported success but wrote no output")
    return secs
