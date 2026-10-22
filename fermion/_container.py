"""Cheap container identification + integrity checks (seek-only, O(1) RAM).

Two defects this module exists to close, both of them in the one command a
user runs to answer *"did my download actually work?"*:

  * **A four-byte question used to cost a whole-file read.** Picking the
    container out of a downloaded directory was `b.read_bytes()[:4] ==
    TRTC_MAGIC` — that materialises the entire file in RAM to look at four
    bytes (measured 524 MB peak for a 300 MB file; ~4 GB for the shipping
    8B, which is the same failure class as the 2026-07-24 Mac OOM).
    `read_magic()` seeks and reads four bytes.

  * **A truncated download reported a clean bill of health.** The header
    lives in the first 60 bytes, so an interrupted download keeps a
    perfectly valid-looking one; `fermion info` printed it and exited 0 on a
    file that was 1.7% of the real container. `check_container()` proves the
    file on disk is the *whole* file before anything is allowed to call it
    good.

The integrity walk reads the 16-byte geometry preamble of each record and
seeks over the payload, so it is O(number of records) — ~250 tiny reads for
the 8B — not O(file size). No weight lane is ever touched and no offset,
record ordering or layout fact is ever printed: callers get a byte count and
a verdict.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

TRTC_MAGIC = b"TRTC"
TINY_MAGICS = (b"QRB8", b"QRL8")

_HASH_CHUNK = 1 << 24


class ContainerError(Exception):
    """A container on disk is not the container it claims to be."""


# --------------------------------------------------------------- identity
def read_magic(path: str | Path, n: int = 4) -> bytes:
    """First `n` bytes of a file. Reads `n` bytes, not the file."""
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def is_trtc(path: str | Path) -> bool:
    return read_magic(path) == TRTC_MAGIC


def find_container(directory: str | Path) -> Path:
    """The TRTC container inside a model directory, found without reading it."""
    directory = Path(directory)
    for b in sorted(directory.glob("*.bin")):
        if is_trtc(b):
            return b
    raise SystemExit(f"{directory}: no TRTC v4 .bin container inside")


# ------------------------------------------------------------- structure
def _loader_bits():
    try:
        from hf_ternary.loader import HEADER, read_header
    except ImportError:
        from fermion._vendor.hf_ternary.loader import HEADER, read_header
    return HEADER, read_header


class _Seeker:
    """Forward-only cursor that refuses to read past EOF."""

    def __init__(self, path: Path, size: int):
        self.f = open(path, "rb")
        self.size = size
        self.off = 0

    def close(self) -> None:
        self.f.close()

    def need(self, n: int, what: str) -> None:
        if self.off + n > self.size:
            raise ContainerError(
                f"container is truncated: {what} needs bytes up to "
                f"{self.off + n:,} but the file is only {self.size:,} bytes "
                f"({self.size / max(1, self.off + n):.1%} of the minimum "
                f"required so far)")

    def take(self, fmt: str, what: str):
        n = struct.calcsize(fmt)
        self.need(n, what)
        self.f.seek(self.off)
        raw = self.f.read(n)
        if len(raw) != n:
            raise ContainerError(f"container is truncated: short read for {what}")
        self.off += n
        return struct.unpack(fmt, raw)

    def skip(self, n: int, what: str) -> None:
        self.need(n, what)
        self.off += n


def expected_size(path: str | Path) -> int:
    """Bytes this container's own structure says it must be.

    Walks the record geometry only. Raises `ContainerError` the moment the
    structure requires a byte the file does not have — which is exactly the
    truncated-download case a header read cannot see.
    """
    path = Path(path)
    HEADER, read_header = _loader_bits()
    h = read_header(path)                       # 60 bytes, validates magic/version
    size = path.stat().st_size
    s = _Seeker(path, size)
    try:
        s.skip(HEADER.size, "header")

        def int8_record(what: str) -> None:
            rows, cols, cols_pad, stride = s.take("<4i", f"{what} geometry")
            if rows <= 0 or cols <= 0 or stride != cols_pad or cols_pad < cols:
                raise ContainerError(f"invalid {what} geometry in container")
            s.skip(rows * stride, f"{what} weights")
            s.skip(rows * 8, f"{what} scales")

        def trit_record(what: str) -> None:
            rows, cols, cols_pad, stride = s.take("<4i", f"{what} geometry")
            if rows <= 0 or cols <= 0 or cols_pad < cols or stride * 8 < cols_pad:
                raise ContainerError(f"invalid {what} geometry in container")
            s.skip(3 * rows * stride, f"{what} weights")
            s.skip(16 * rows, f"{what} scales")
            (has_bias,) = s.take("<i", f"{what} bias flag")
            if has_bias:
                s.skip(4 * rows, f"{what} bias")

        if h.eok == 2:
            int8_record("input embedding")
        norm_floats = {1: 2 * h.hidden,
                       2: 2 * h.hidden + h.hidden + h.intermediate,
                       3: 2 * h.hidden + 2 * h.head_dim}[h.arch]
        for li in range(h.n_layers):
            s.skip(norm_floats * 4, f"layer {li} norms")
            for pk in ("q", "k", "v", "o", "gate", "up", "down"):
                trit_record(f"layer {li} {pk} projection")
        s.skip(h.hidden * 4, "final norm")
        int8_record("output embedding")
        return s.off
    finally:
        s.close()


# -------------------------------------------------------------- checksums
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def manifest_pin(container: str | Path) -> dict | None:
    """`{bytes, sha256, source}` for this container from a sibling MANIFEST.json.

    The published model repo ships MANIFEST.json next to the container, so a
    user who downloaded the model has a signed length + digest on disk and we
    should use it rather than trusting the transfer.
    """
    container = Path(container)
    mf = container.parent / "MANIFEST.json"
    if not mf.is_file():
        return None
    try:
        doc = json.loads(mf.read_text())
    except (OSError, ValueError):
        return None
    wc = doc.get("weights_container") or {}
    if wc.get("file") == container.name and wc.get("sha256"):
        return {"bytes": wc.get("bytes"), "sha256": wc["sha256"],
                "source": "MANIFEST.json weights_container"}
    entry = (doc.get("files") or {}).get(container.name)
    if isinstance(entry, dict) and entry.get("sha256"):
        return {"bytes": entry.get("bytes"), "sha256": entry["sha256"],
                "source": f"MANIFEST.json files[{container.name}]"}
    return None


# ------------------------------------------------------------- public API
def check_container(path: str | Path, *, checksum: bool = True,
                    on_note=None) -> dict:
    """Prove a TRTC container is complete. Raises `ContainerError` if not.

    Checks, cheapest first:
      1. the file length equals the length its own record structure implies;
      2. the length matches a MANIFEST.json pin, when one is on disk;
      3. the sha256 matches that pin (opt-out via `checksum=False`).

    Returns a dict of what was checked so the caller can report it.
    """
    path = Path(path)
    size = path.stat().st_size
    out: dict = {"bytes": size, "structural_bytes": None, "pin": None,
                 "sha256": None, "checked": []}

    want = expected_size(path)
    out["structural_bytes"] = want
    if want != size:
        raise ContainerError(
            f"container length is wrong: the file is {size:,} bytes but its "
            f"own structure describes {want:,} bytes "
            f"({size - want:+,}). The file is incomplete or corrupt.")
    out["checked"].append("structure")

    pin = manifest_pin(path)
    if pin:
        out["pin"] = pin
        if pin.get("bytes") is not None and pin["bytes"] != size:
            raise ContainerError(
                f"container length is wrong: {size:,} bytes on disk but "
                f"{pin['bytes']:,} bytes pinned in {pin['source']}.")
        out["checked"].append("manifest-bytes")
        if checksum:
            if on_note:
                on_note(f"hashing {size / 2**30:.2f} GiB against {pin['source']}…")
            got = sha256_file(path)
            out["sha256"] = got
            if got != pin["sha256"]:
                raise ContainerError(
                    f"container checksum does not match: sha256 {got} on disk "
                    f"but {pin['sha256']} pinned in {pin['source']}. "
                    f"The bytes are not the bytes we published.")
            out["checked"].append("manifest-sha256")
    return out


# ------------------------------------------------------- FV5 GGUF provenance
GGUF_MAGIC = 0x46554747


def read_gguf_kv(path, wanted):
    """Return the string value of `wanted` from a GGUF's KV block, or None.

    Parses the header and KV pairs ONLY. It never reads the tensor-info
    section, which is exactly the part that raises on an unknown weight type --
    the same failure a stock reader hits on any file using ids it does not
    know. So this works on files we cannot otherwise load, which is the point.
    """
    import struct
    T_STR, T_ARR = 8, 9
    fixed = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
             6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    with open(path, "rb") as f:
        head = f.read(24)
        if len(head) != 24:
            return None
        magic, _ver, _nt, n_kv = struct.unpack("<IIQQ", head)
        if magic != GGUF_MAGIC:
            return None

        def rd(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def rstr():
            return f.read(rd("<Q")).decode("utf-8", "replace")

        def rval(t):
            if t == T_STR:
                return rstr()
            if t == T_ARR:
                et = rd("<I")
                n = rd("<Q")
                return [rval(et) for _ in range(n)]
            return rd(fixed[t])

        for _ in range(n_kv):
            k = rstr()
            v = rval(rd("<I"))
            if k == wanted:
                return v
    return None


LLAMA_FTYPE_MOSTLY_FV5 = 42


def check_gguf_provenance(path, allow_legacy=False):
    """Raise unless an FV5 pack carries our provenance key.

    Scoped to FV5 files ONLY, decided from `general.file_type` in the KV block.
    A first cut of this checked every GGUF and would therefore have refused
    every third-party model a user pointed at a bundled runner -- the guard is
    about OUR weight-type numbers being reused, so a file that does not claim
    them is none of its business. Returns None for those, unchanged.
    """
    from fermion._brand import (GGUF_PROVENANCE_KEY, GGUF_PROVENANCE_PREFIX,
                                LEGACY_GGUF_MESSAGE)
    ftype = read_gguf_kv(path, "general.file_type")
    if ftype != LLAMA_FTYPE_MOSTLY_FV5:
        return None
    prov = read_gguf_kv(path, GGUF_PROVENANCE_KEY)
    ok = isinstance(prov, str) and prov.startswith(GGUF_PROVENANCE_PREFIX)
    if ok or allow_legacy:
        return prov
    raise ContainerError(LEGACY_GGUF_MESSAGE)
