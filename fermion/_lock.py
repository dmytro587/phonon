"""A cross-process lock for the model cache (0.1.25).

Two `fermion` processes starting on a fresh machine used to unpack the same
model into the same directory at the same time; the second one saw a
half-written tree and died with `FileNotFoundError: ... model-00001.safetensors`
(reports/pip_linux_import_repro_20260906.md §7). `fetch.ensure()` now takes
this lock around download + verify + unpack, re-checks the cache once it holds
the lock, and the unpack itself still lands through one atomic rename — so the
second process simply waits and then finds the finished model.

`fcntl.flock` on POSIX, `msvcrt.locking` on Windows; the lock file is a tiny
sibling of the model directory and is left in place. If the filesystem cannot
lock (some network mounts), the lock degrades to a note and the download
proceeds unlocked, which is exactly 0.1.24's behaviour.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


class CacheLock:
    """`with CacheLock(path, what="the Phonon-1 model"): ...`"""

    #: How long to wait quietly before saying that another process holds it.
    QUIET_S = 1.0

    def __init__(self, path, *, what: str = "this model", note=None):
        self.path = Path(path)
        self.what = what
        self._note = note or (lambda m: print(m, file=sys.stderr, flush=True))
        self._fd = None
        self.waited = 0.0          # seconds spent waiting for another holder
        self.locked = False

    # ---------------------------------------------------------------- posix
    def _try_posix(self, blocking: bool) -> bool:
        import fcntl
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(self._fd, flags)
            return True
        except BlockingIOError:
            return False

    # -------------------------------------------------------------- windows
    def _try_windows(self, blocking: bool) -> bool:
        import msvcrt
        while True:
            try:
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                if not blocking:
                    return False
                time.sleep(0.2)

    def _try(self, blocking: bool) -> bool:
        if os.name == "nt":
            return self._try_windows(blocking)
        return self._try_posix(blocking)

    def __enter__(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as exc:
            self._note(f"[fermion] could not create the cache lock "
                       f"{self.path} ({exc}); continuing without it")
            self._fd = None
            return self
        try:
            if self._try(blocking=False):
                self.locked = True
                return self
            t0 = time.perf_counter()
            # Say so once, after a short quiet period, then wait as long as it
            # takes: the holder is downloading and unpacking hundreds of MB.
            deadline = t0 + self.QUIET_S
            while time.perf_counter() < deadline:
                if self._try(blocking=False):
                    self.locked = True
                    self.waited = time.perf_counter() - t0
                    return self
                time.sleep(0.1)
            self._note(f"[fermion] another fermion process is downloading "
                       f"{self.what}; waiting for it to finish")
            self._try(blocking=True)
            self.locked = True
            self.waited = time.perf_counter() - t0
            return self
        except OSError as exc:
            self._note(f"[fermion] the cache lock is not supported here "
                       f"({exc}); continuing without it")
            self._close()
            return self

    def __exit__(self, *exc):
        if self._fd is not None and self.locked:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
        self._close()
        return False

    def _close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        self.locked = False
