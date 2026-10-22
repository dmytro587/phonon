"""Native decode backend: drive the prebuilt `fermion-run-<platform>` runtime.

WHY THIS FILE EXISTS
--------------------
Until 0.1.4 the pip CLI downloaded `bin/fermion-run-*` from the model repo
(it is in `HF_RUNTIME_PATTERNS`) and then never executed it: every token came
out of the vendored torch reference loader. Measured on an M5 MacBook with
the 0.6B-Chat container that is ~1.2 tok/s, while the same container through
`bin/fermion-run-macos-arm64` runs at 150-190 tok/s. A first-time user's
`pip install fermion && fermion chat` was therefore two orders of magnitude
slower than the number on the model card, which was taken with the binary.

This module makes the binary the DEFAULT decode path when one is present for
the running platform, and leaves the torch path as the fallback (and as the
reference the native path is gated against).

WHAT THE BINARY ACTUALLY TAKES (the runner's own usage text)
------------------------------------------------------------
    fermion-run-<plat> <container.bin> "<id0,id1,...>" <n_new> <n_threads> \
        [logits_out.bin] [--temp T] [--top-p P] [--rep-pen R] [--seed S] \
        [--stop-id I0[,I1...]] [--min-p F] [--presence-pen F] [--freq-pen F] \
        [--dry M [--dry-base B --dry-allow N]] [--pen-window N] \
        [--eos-exempt-minp] [--eos-boost B:N] [--ppl]

Pre-tokenized ids in, ids out. stdout is one comma-separated id line followed
by `tok/s: ...`; with `TRIT_STREAM=1` in the environment it additionally
prints `READY` before prefill and `TOK <id>` per token as they are produced,
which is what gives this backend real token streaming (used below).

THE TEMPERATURE-0 QUIRK, AND HOW GREEDY IS MAPPED
--------------------------------------------------
At `--temp 0` the runtime takes its pure argmax fast path and every sampler
flag (`--rep-pen`, `--top-p`, `--min-p`,
the penalty window, ...) is silently ignored. So a naive "temperature 0 ->
--temp 0" mapping would DROP a repetition penalty the caller asked for, which
is precisely the bug the serve lane fixed on the torch side (where a greedy
`do_sample=False` still runs the logits processors).

Mapping used here, so determinism AND the penalty both survive:

  temperature == 0, rep_penalty == 1.0
      -> `--temp 0`. Pure argmax. Nothing to preserve, and this is the exact
         configuration every banked greedy receipt was measured at.
  temperature == 0, rep_penalty != 1.0
      -> `--temp 0.01 --top-p 1.0 --min-p 0.999 --rep-pen R --pen-window W
          --seed 0`. Sampling is switched on so the penalty is applied at
         all, then the candidate set is collapsed back to the argmax:
         `--min-p 0.999` culls every token whose probability is under
         0.999x the max, and at T=0.01 a logit gap of even 0.05 is a
         probability ratio of e^5, so only the argmax (or an exact tie)
         survives. Fixed seed 0 => byte-reproducible.
         `--min-p` additionally flips the runtime onto its modern sampler
         stack, whose penalty window honours `--pen-window` (the legacy
         path hardcodes a 64-token window regardless of the flag). That is
         what makes
         this mapping match the torch path's windowed penalty as well.
  temperature > 0
      -> `--temp T --top-p P [--rep-pen R --pen-window W] --seed S`, i.e.
         the graded shipping configuration verbatim, legacy sampler path,
         which is the path the C sampler receipts were graded on.

TOKEN IDENTITY
--------------
Greedy output through this backend is gated token-for-token against the torch
reference path. The runtime computes in fp32; the CLI's torch default is
bfloat16, which is a coarser accumulator, so the identity statement is
native == torch@float32.
"""
from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# Platform -> the binary name the model repo publishes under bin/.
# Only the entries here are ever executed; anything else falls back to torch.
PLATFORM_BINARIES = {
    ("Darwin", "arm64"): "fermion-run-macos-arm64",
    ("Darwin", "aarch64"): "fermion-run-macos-arm64",
    ("Linux", "x86_64"): "fermion-run-linux-x64",
    ("Linux", "amd64"): "fermion-run-linux-x64",
}

# Greedy-with-penalty emulation constants (see the module docstring).
GREEDY_TEMP = 0.01
GREEDY_MIN_P = 0.999
GREEDY_SEED = 0

_ID_LINE = re.compile(r"^[0-9]+(,[0-9]+)*$")
_TOKS = re.compile(r"tok/s: ([0-9.]+)")


class NativeUnavailable(Exception):
    """No usable native runtime for this platform/container/request."""


class SessionError(Exception):
    """The persistent session process failed; callers fall back to one-shot."""


# --------------------------------------------------------------- discovery
def platform_tag() -> str | None:
    """The binary basename this machine can run, or None."""
    return PLATFORM_BINARIES.get((platform.system(), platform.machine().lower()))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _sidecar_ok(binary: Path) -> tuple[bool, str]:
    """Verify `<binary>.sha256` when the repo shipped one.

    The model repo publishes a standard `shasum -a 256` line next to every
    binary. We are about to *execute* this file, so when the pin is on disk
    it is checked; a mismatch means the bytes are not the bytes we published
    and the native path is refused (torch fallback), never silently run.
    """
    pin = binary.with_name(binary.name + ".sha256")
    if not pin.is_file():
        return True, "no .sha256 sidecar on disk (not verified)"
    try:
        want = pin.read_text().split()[0].strip().lower()
    except (OSError, IndexError):
        return True, "unreadable .sha256 sidecar (not verified)"
    got = _sha256(binary)
    if got != want:
        return False, f"sha256 {got[:12]}… != pinned {want[:12]}…"
    return True, f"sha256 verified against sidecar ({got[:12]}…)"


def find_binary(container: Path, *, explicit: str | None = None
                ) -> tuple[Path | None, str]:
    """(binary, why). `binary` is None when the torch path must be used.

    Search order:
      1. $FERMION_NATIVE_BIN (an explicit path, for developers/CI)
      2. <model dir>/bin/<tag>       - the published HF repo layout
      3. <model dir>/../bin/<tag>    - container one level down (gguf/, mlx/)

    Any `.gguf` handed to a bundled runner is checked for the Fermion
    provenance key first (see fermion._container.check_gguf_provenance): the
    runner would otherwise load a foreign file that reuses our weight-type
    numbers and produce fluent nonsense with no error.
      4. <tag> on $PATH
    """
    explicit = explicit or os.environ.get("FERMION_NATIVE_BIN")
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p, f"FERMION_NATIVE_BIN={p}"
        return None, f"FERMION_NATIVE_BIN={p} does not exist"

    tag = platform_tag()
    if not tag:
        return None, (f"no native runtime is published for "
                      f"{platform.system()}/{platform.machine()}")

    # NOT .resolve(): `snapshot_download` materialises every file in
    # snapshots/<rev>/ as a SYMLINK into blobs/<sha>, so resolving the
    # container walks us out of the repo tree into a flat blob store where
    # bin/ does not exist. The directory the user (or the hub) laid the model
    # out in is the one that has bin/ next to it.
    root = Path(container).absolute().parent
    seen = []
    for base in (root, root.parent, Path(container).resolve().parent):
        cand = base / "bin" / tag
        if cand in seen:
            continue
        seen.append(cand)
        if cand.is_file():
            return cand, f"found {cand}"
    onpath = shutil.which(tag)
    if onpath:
        return Path(onpath), f"found {onpath} on PATH"
    return None, (f"{tag} not found under {root / 'bin'} "
                  f"(re-download the model repo to get bin/, or point "
                  f"$FERMION_NATIVE_BIN at a copy)")


def _make_executable(binary: Path) -> None:
    """Restore the +x bit the Hub client throws away. Never skip this.

    The Hub stores these files 100755, but `hf_hub_download` lands them
    **0644**: `file_download._chmod_and_move()` (huggingface_hub 1.25.1,
    file_download.py:1964-1990) deliberately discards the repo's file mode and
    chmods every blob to the mode of a freshly `touch`ed file. So on a plain
    `pip install fermion && fermion chat`, the runtime we just downloaded is
    not executable and `exec` returns EACCES — on every platform, on every
    user's first run. Measured by the certification gates: the mode is 0644
    after `resolve_model` and must be executable after the first decode.

    `Path.chmod` follows symlinks, which is what we want: inside a snapshot
    the runner is a symlink into `blobs/`, and the blob is the file `exec`
    ultimately opens.
    """
    mode = binary.stat().st_mode
    if not mode & 0o111:
        try:
            binary.chmod(mode | 0o755)
        except OSError as exc:                              # pragma: no cover
            raise NativeUnavailable(
                f"cannot chmod +x {binary} ({exc}). The Hub client downloads "
                f"the runtime without its executable bit; run "
                f"`chmod +x {binary}` and retry.")


def _seed() -> int:
    """Sampling seed: `$FERMION_SEED` if pinned, else a fresh one per call."""
    env = os.environ.get("FERMION_SEED")
    if env and env.lstrip("-").isdigit():
        return int(env)
    import random
    return random.randrange(1, 2 ** 31)


# Hard ceiling on the thread count this package will ever hand the native
# runtime. INDEPENDENT of any clamp inside the C runtime: on 2026-07-27 the
# runtime overflowed a per-thread heap buffer above 16 threads (SIGSEGV on the
# 8B, silently WRONG TOKENS at exit code 0 on the 0.6B) and `_threads()`
# happily asked for 23 on a 24-core Linux box. The runtime is fixed, but the
# lesson is that the Python side must not be able to trigger a runtime
# regression on its own. 16 is where every measurement we have is clean,
# and decode is flat well before it
# (0.6B: 41.5 tok/s at 16 vs 42.4 at 23 on the Linux AVX2 venue), so the cap
# costs approximately nothing.
#
# NOTE: `os.cpu_count()` is NOT trustworthy here and neither is
# `sched_getaffinity` -- on the Modal container that found the bug BOTH
# reported 24 while the cgroup allowed 8, and `/sys/fs/cgroup/cpu.max` did not
# exist. Cgroup awareness would not have prevented this. A constant ceiling
# would.
#
# `$FERMION_MAX_THREADS` raises the ceiling for deliberate experiments (the
# internal thread-sweep gate drives the binary directly instead, precisely so
# it can test past this cap).
THREAD_CAP = 16


def _thread_cap() -> int:
    env = os.environ.get("FERMION_MAX_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return THREAD_CAP


def _threads() -> int:
    """Decode threads, `$FERMION_THREADS` wins, `THREAD_CAP` wins over both.

    Measured on the M5 (4P+6E, 10 logical), greedy, 64 tokens, best of
    several — decode tok/s by thread count:

        0.6B-Chat   2:142   4:197   6:209   7:239   8:223-245   9:227   10:150
        8B ship     6:22.8          8:21.4                      9:24.4  10:17.1

    Two things fall out. Neither "P-cores only" (4) nor "every core" (10) is
    right: the runtime's worker threads synchronise every step, so
    oversubscribing the machine makes every thread wait on a
    descheduled peer and costs 30-40%. And the 0.6B is flat from 6 to 9 while
    the 8B is not — 9 is where the 8B reaches 24.4 tok/s, which reproduces
    the published receipt (24.94 at 9 threads) and the
    number on the model card. So `cpu_count - 1`: leave one core for the OS
    and this Python process, and reproduce the published measurement on the
    machine it was published from. `$FERMION_THREADS` overrides.

    ...and then everything is capped at `THREAD_CAP` (see above). On a 24-core
    Linux server `cpu_count - 1` is 23, which is how a heap overflow in the
    runtime reached users as silently wrong text. Apple silicon never hit it
    because M-series parts top out at 10-16 logical cores, so `cpu_count - 1`
    was always <= 15 and every Mac test passed.
    """
    cap = _thread_cap()
    env = os.environ.get("FERMION_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return min(int(env), cap)
    n = os.cpu_count() or 4
    return min(max(1, n - 1) if n >= 4 else n, cap)


def _header(container: Path):
    """TRTC v4 header without importing torch (`struct`, 60 bytes)."""
    import struct
    raw = open(container, "rb").read(60)
    if len(raw) != 60 or raw[:4] != b"TRTC":
        raise NativeUnavailable(f"{container}: not a TRTC container")
    h = struct.unpack("<Iii9i2fi", raw)
    if h[1] != 4 or h[2] not in (1, 2, 3):
        raise NativeUnavailable(
            f"{container}: native runtime needs TRTC v4 arch 1/2/3, "
            f"got v{h[1]} arch {h[2]}")
    # dims are h[3..11]: n_layers hidden n_heads n_kv_heads head_dim
    # rotary_dims intermediate vocab max_pos. (0.1.10 also FIXES "vocab":
    # it used to read h[12], which is ln_eps -- a float32 1e-6 masquerading
    # as the vocab size in `config.vocab_size` since 0.1.4.)
    return {"version": h[1], "arch": h[2], "n_layers": h[3], "hidden": h[4],
            "n_heads": h[5], "n_kv_heads": h[6], "head_dim": h[7],
            "vocab": h[10], "max_pos": h[11]}


def kv_kib_per_token(header: dict, kv_dtype: str = "fp16") -> float:
    """KV-cache cost per cached position, in KiB (k + v, all layers).

    fp32 = 4 B/elt, fp16 = 2 B/elt, int8 = 1 B/elt + one fp32 scale per
    stored head-vector (matches the C runtime's arithmetic exactly).
    """
    L, kv, hd = header["n_layers"], header["n_kv_heads"], header["head_dim"]
    if kv_dtype == "fp16":
        per_vec = hd * 2.0
    elif kv_dtype == "int8":
        per_vec = hd * 1.0 + 4.0            # int8 codes + one fp32 scale
    else:
        per_vec = hd * 4.0
    return (L * kv * 2 * per_vec) / 1024.0  # k + v


def _available_ram_bytes() -> int | None:
    """Best-effort available (not total) physical RAM; None when unknowable.

    Used only to make the KV-cache arithmetic loud BEFORE a resident runner
    OOMs/jetsams the machine -- a wrong None here just skips the warning.
    """
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        elif sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                                 timeout=5).stdout
            page = 16384
            m = re.search(r"page size of (\d+) bytes", out)
            if m:
                page = int(m.group(1))
            pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative",
                        "Pages purgeable"):
                m = re.search(rf"{key}:\s+(\d+)", out)
                if m:
                    pages += int(m.group(1))
            if pages:
                return pages * page
    except Exception:                                       # pragma: no cover
        pass
    return None


# ------------------------------------------------------------------ engine
class NativeModel:
    """Stand-in for the torch model that decodes through the C runtime.

    Deliberately duck-types only the handful of attributes the rest of the
    package touches on a loaded model (`config.vocab_size`, `eval()`,
    `to()`), so `chat`, `generate`, `serve` and `bench` need no branching
    beyond the one dispatch in `cli.iter_generate`.
    """

    backend = "native"

    def __init__(self, binary: Path, container: Path, *, threads: int | None = None,
                 note: str = "", kv_dtype: str = "fp16",
                 yarn_factor: float | None = None,
                 yarn_orig_max: int | None = None,
                 draft_path: Path | None = None):
        self.binary = Path(binary)
        self.container = Path(container)
        self.threads = threads or _threads()
        self.note = note
        _make_executable(self.binary)
        ok, why = _sidecar_ok(self.binary)
        if not ok:
            raise NativeUnavailable(
                f"{self.binary}: {why}. Refusing to execute it; "
                f"delete bin/ and re-download the model.")
        self.integrity = why
        self.header = _header(self.container)
        # 0.1.10 runtime flags: fp16 KV storage + YaRN rope, passed to every
        # runner invocation (one-shot AND session). An older runner binary
        # rejects them with "unknown arg" -> NativeUnavailable/SessionError,
        # never a silent ignore.
        if kv_dtype not in ("fp32", "fp16", "int8"):
            raise NativeUnavailable(f"kv_dtype must be fp16 (default), fp32 "
                                    f"or int8, got {kv_dtype!r}")
        self.kv_dtype = kv_dtype
        self.yarn_factor = float(yarn_factor) if yarn_factor else None
        self.yarn_orig_max = int(yarn_orig_max) if yarn_orig_max else None
        if self.yarn_factor is not None and self.yarn_factor < 1.0:
            raise NativeUnavailable("--yarn-factor must be >= 1.0")
        self.runtime_flags: list[str] = []
        # Always pass the KV dtype explicitly (0.1.10): the wheel ships the
        # matching 0.1.10 runner (default fp16), and being explicit keeps the
        # wheel self-documenting and robust regardless of a binary's own
        # default. fp16 is passed too so a resident runner logs it.
        self.runtime_flags += ["--kv-dtype", self.kv_dtype]
        if self.yarn_factor is not None:
            self.runtime_flags += ["--yarn-factor", repr(self.yarn_factor)]
            if self.yarn_orig_max:
                self.runtime_flags += ["--yarn-orig-max",
                                       str(self.yarn_orig_max)]
        # effective context window: model max_pos, or factor x orig when YaRN
        # extends it (mirrors the C runtime's effective-context rule)
        self.context = int(self.header["max_pos"])
        if self.yarn_factor is not None and self.yarn_factor > 1.0:
            orig = self.yarn_orig_max or self.header["max_pos"]
            self.context = int(self.yarn_factor * orig)
        # 0.1.14: native speculative decoding. The runner's --draft is
        # greedy-only (the greedy-exact acceptance rule); argv() attaches it
        # on greedy calls and quietly runs plain for sampled ones (the
        # drafting note the CLI prints already states the greedy scoping).
        self.draft_path = Path(draft_path) if draft_path else None
        self._session = None          # NativeSession when enable_session() ran
        self._session_ctx = 0
        self.session_note = "session not enabled"
        self.last_reused = None       # prompt positions served from the KV cache
        # per-process session/reuse counters, surfaced by `fermion serve`'s
        # /health (0.1.8): silent 2-minute agent turns were traced to every
        # request quietly falling back to one-shot mode —
        # fallbacks must be LOUD and countable.
        self.session_stats = {
            "requests": 0,            # requests routed while a session was armed
            "served_by_session": 0,   # ... of which the resident runner served
            "fallbacks": 0,           # ... of which fell back to one-shot
            "last_fallback": None,    # human-readable reason for the last one
            "last_request": None,     # {"prompt_tokens", "reused", "prefilled"}
        }
        from types import SimpleNamespace
        self.config = SimpleNamespace(vocab_size=self.header["vocab"],
                                      model_type="trtc_v4",
                                      max_position_embeddings=self.context)

    # -- torch-model shims ------------------------------------------------
    def eval(self):
        return self

    def to(self, *_a, **_kw):
        return self

    def __repr__(self) -> str:                              # pragma: no cover
        return (f"<NativeModel {self.binary.name} threads={self.threads} "
                f"container={self.container.name}>")

    def describe(self) -> dict:
        return {"backend": "native", "binary": str(self.binary),
                "binary_name": self.binary.name, "threads": self.threads,
                "integrity": self.integrity,
                "platform": f"{platform.system()}/{platform.machine()}",
                "session": bool(self._session and self._session.alive()),
                "session_ctx": self._session_ctx or None,
                "session_note": self.session_note,
                "session_stats": dict(self.session_stats),
                "kv_dtype": self.kv_dtype,
                "kv_kib_per_token": round(
                    kv_kib_per_token(self.header, self.kv_dtype), 1),
                "yarn_factor": self.yarn_factor,
                "context": self.context}

    # -- sampler mapping --------------------------------------------------
    def argv(self, ids, *, max_new: int, temperature: float, top_p: float,
             rep_penalty: float, pen_window: int, stop_ids) -> list[str]:
        cmd = [str(self.binary), str(self.container), ",".join(map(str, ids)),
               str(int(max_new)), str(self.threads)]
        penalised = bool(rep_penalty) and float(rep_penalty) != 1.0
        if temperature > 0:
            # Fresh seed per call unless $FERMION_SEED pins one. The torch
            # path samples from the process RNG, so two identical sampled
            # requests there give two different answers; a hardcoded seed
            # here would make the native path quietly deterministic instead,
            # which is a different product, not a faster one.
            cmd += ["--temp", repr(float(temperature)),
                    "--top-p", repr(float(top_p)),
                    "--seed", str(_seed())]
            if penalised:
                cmd += ["--rep-pen", repr(float(rep_penalty)),
                        "--pen-window", str(int(pen_window))]
        elif penalised:
            # greedy REQUESTED but a penalty is set: the runtime would drop
            # the penalty at --temp 0, so emulate argmax on top of sampling.
            cmd += ["--temp", repr(GREEDY_TEMP), "--top-p", "1.0",
                    "--min-p", repr(GREEDY_MIN_P),
                    "--rep-pen", repr(float(rep_penalty)),
                    "--pen-window", str(int(pen_window)),
                    "--seed", str(GREEDY_SEED)]
        else:
            cmd += ["--temp", "0"]                       # pure argmax
            if self.draft_path is not None:
                # 0.1.14: native speculative decoding on the pure-greedy
                # path only (the runner's --draft contract; sampled and
                # penalty-emulated calls run plain).
                cmd += ["--draft", str(self.draft_path)]
        if stop_ids:
            cmd += ["--stop-id", ",".join(str(int(s)) for s in stop_ids[:16])]
        cmd += self.runtime_flags          # 0.1.10: --kv-dtype / --yarn-*
        return cmd

    # -- persistent session (0.1.7) ----------------------------------------
    def enable_session(self, ctx: int = 8192) -> bool:
        """Arm the persistent runner: ONE resident process holds the mmap'd
        weights and the KV cache across requests, so each request prefills
        only the prompt tail the cache has not seen. This is the fix for the
        agent-loop cost model measured with an MCP-based agent framework (a
        fixed ~4.3k token preamble was re-prefilled from zero, plus a ~6 s
        process start,
        on EVERY step). Returns True when the session is up; False (with
        `session_note` saying why) keeps the one-shot-per-request behaviour —
        which is also what a pre-0.1.7 runner binary that does not know
        `--session` degrades to. `$FERMION_SESSION=0` disables;
        `$FERMION_SESSION_CTX` overrides the KV capacity in positions
        (KV RAM scales linearly with it: ~288 KiB/position on the 8B).
        """
        env = os.environ.get("FERMION_SESSION")
        if env is not None and env.strip() == "0":
            self.session_note = "disabled by FERMION_SESSION=0"
            return False
        if self.draft_path is not None:
            # 0.1.14: the resident-session protocol has no draft leg yet;
            # drafted requests run one-shot (loud, countable, correct).
            self.session_note = ("sessions disabled while --draft is armed "
                                 "(0.1.14: native drafting is one-shot)")
            return False
        ctx_env = os.environ.get("FERMION_SESSION_CTX")
        if ctx_env and ctx_env.isdigit():
            ctx = int(ctx_env)
        ctx = min(int(ctx), self.context)
        # 0.1.10 RAM guard: state the arithmetic BEFORE a resident KV cache
        # that cannot fit physical memory jetsams the machine. tokens x
        # KiB/token vs what the OS says is available right now.
        kib = kv_kib_per_token(self.header, self.kv_dtype)
        need = int(ctx * kib * 1024)
        avail = _available_ram_bytes()
        if avail is not None and need > avail:
            fp16_hint = ("" if self.kv_dtype == "fp16" else
                         ", or pass --kv-dtype fp16 (halves KV bytes)")
            self.session_note = (
                f"session REFUSED: KV cache {ctx} positions x {kib:.0f} "
                f"KiB/token ({self.kv_dtype}) = {need / 2**30:.2f} GiB > "
                f"{avail / 2**30:.2f} GiB available RAM. Lower "
                f"--session-ctx{fp16_hint}.")
            print(f"[fermion] {self.session_note}", file=sys.stderr)
            return False
        if avail is not None and need > 0.8 * avail:
            print(f"[fermion] WARNING: KV cache {ctx} positions x {kib:.0f} "
                  f"KiB/token ({self.kv_dtype}) = {need / 2**30:.2f} GiB is "
                  f">80% of the {avail / 2**30:.2f} GiB available RAM -- "
                  f"expect swapping. Lower --session-ctx"
                  + ("" if self.kv_dtype == "fp16"
                     else " or pass --kv-dtype fp16") + ".",
                  file=sys.stderr)
        try:
            self._session = NativeSession(self.binary, self.container,
                                          self.threads, ctx,
                                          runtime_flags=self.runtime_flags)
            self._session_ctx = self._session.ctx
            self.session_note = (f"session up: resident runner, KV ctx "
                                 f"{self._session.ctx} positions")
            return True
        except SessionError as exc:
            self._session = None
            self._session_ctx = 0
            self.session_note = (f"session unavailable ({exc}); one-shot "
                                 f"mode per request")
            return False

    def close_session(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _session_line_params(self, *, temperature: float, top_p: float,
                             rep_penalty: float, pen_window: int) -> tuple:
        """(temp, topp, reppen, minp, penwin, seed) for a session request —
        the exact sampler mapping `argv()` uses for the one-shot runner (see
        the module docstring), expressed as protocol fields. Fields the
        one-shot argv would omit are sent as the runtime's defaults, so the
        C-side sampler state is identical either way (gated
        byte-identical)."""
        penalised = bool(rep_penalty) and float(rep_penalty) != 1.0
        if temperature > 0:
            return (float(temperature), float(top_p),
                    float(rep_penalty) if penalised else 1.0,
                    0.0, int(pen_window), _seed())
        if penalised:   # greedy + penalty: argmax-on-top-of-sampling emulation
            return (GREEDY_TEMP, 1.0, float(rep_penalty), GREEDY_MIN_P,
                    int(pen_window), GREEDY_SEED)
        return (0.0, 0.9, 1.0, 0.0, int(pen_window), GREEDY_SEED)  # pure argmax

    # -- decode -----------------------------------------------------------
    def stream_ids(self, ids, *, max_new: int, temperature: float, top_p: float,
                   rep_penalty: float, pen_window: int, stop_ids,
                   should_stop=None):
        """Yield generated token ids one at a time, as the runtime emits them.

        Routed through the persistent session when one is armed (KV-prefix
        reuse + no per-request process start). A session failure BEFORE the
        first token falls back to the one-shot runner for this request and the
        session respawns lazily on the next; a mid-stream death propagates as
        NativeUnavailable exactly like the one-shot path (see below).
        """
        self.last_reused = None
        fallback_reason = None
        if self._session is not None or self._session_ctx:
            self.session_stats["requests"] += 1
            need = len(ids) + int(max_new) + 1
            try:
                if self._session is None or not self._session.alive():
                    self._session = NativeSession(self.binary, self.container,
                                                  self.threads,
                                                  self._session_ctx or 8192,
                                                  runtime_flags=self.runtime_flags)
                    self._session_ctx = self._session.ctx
                if need <= self._session.ctx:
                    yield from self._session_stream(
                        ids, max_new=max_new, temperature=temperature,
                        top_p=top_p, rep_penalty=rep_penalty,
                        pen_window=pen_window, stop_ids=stop_ids,
                        should_stop=should_stop)
                    self.session_stats["served_by_session"] += 1
                    return
                fallback_reason = (
                    f"prompt {len(ids)} tok + max_tokens {int(max_new)} + 1 "
                    f"= {need} positions > session ctx {self._session.ctx}")
                # ONE loud line, with the numbers and the fix. This exact
                # shape caused silent ~2-minute turns: agent frameworks ask
                # for huge max_tokens, prompt+ceiling quietly exceeded the
                # session ctx on EVERY request, and each step re-prefilled
                # the whole preamble in one-shot mode.
                print(f"[fermion] SESSION FALLBACK -> one-shot (FULL prefill, "
                      f"no KV reuse) because {fallback_reason}. Fix: raise "
                      f"--session-ctx and/or lower --max-new-ceiling so that "
                      f"prompt + max_tokens + 1 <= session ctx.",
                      file=sys.stderr)
            except SessionError as exc:
                fallback_reason = f"session failed before first token ({exc})"
                print(f"[fermion] SESSION FALLBACK -> one-shot (FULL prefill, "
                      f"no KV reuse) because {fallback_reason}; the session "
                      f"respawns on the next request", file=sys.stderr)
                self.close_session()
            self.session_stats["fallbacks"] += 1
            self.session_stats["last_fallback"] = fallback_reason
            self.session_stats["last_request"] = {
                "prompt_tokens": len(ids), "reused": 0, "prefilled": len(ids),
                "via": "one-shot fallback"}
        yield from self._stream_ids_oneshot(
            ids, max_new=max_new, temperature=temperature, top_p=top_p,
            rep_penalty=rep_penalty, pen_window=pen_window,
            stop_ids=stop_ids, should_stop=should_stop)
        if fallback_reason is not None:
            # keep the per-request reuse log consistent across BOTH paths
            print(f"[fermion] session: 0/{len(ids)} prompt positions served "
                  f"from the KV cache, {len(ids)} prefilled (one-shot "
                  f"fallback)", file=sys.stderr)

    def _session_stream(self, ids, *, max_new, temperature, top_p, rep_penalty,
                        pen_window, stop_ids, should_stop):
        temp, topp, reppen, minp, penwin, seed = self._session_line_params(
            temperature=temperature, top_p=top_p, rep_penalty=rep_penalty,
            pen_window=pen_window)
        yield from self._session.request(
            ids, n_new=int(max_new), temp=temp, topp=topp, reppen=reppen,
            minp=minp, penwin=penwin, seed=seed,
            stop_ids=[int(s) for s in (stop_ids or [])][:16],
            should_stop=should_stop)
        self.last_tok_s = self._session.last_tok_s
        self.last_reused = self._session.last_reused
        if self.last_reused is not None:
            self.session_stats["last_request"] = {
                "prompt_tokens": len(ids), "reused": self.last_reused,
                "prefilled": len(ids) - self.last_reused, "via": "session"}
            print(f"[fermion] session: {self.last_reused}/{len(ids)} prompt "
                  f"positions served from the KV cache, "
                  f"{len(ids) - self.last_reused} prefilled", file=sys.stderr)

    def _stream_ids_oneshot(self, ids, *, max_new: int, temperature: float,
                            top_p: float, rep_penalty: float, pen_window: int,
                            stop_ids, should_stop=None):
        """One runner process per request — the fallback path.

        `TRIT_STREAM=1` makes the runtime print `TOK <id>` per token; without
        it we would have to wait for the whole completion before showing the
        first character, which for a chat REPL is the difference between a
        live typewriter and a stall.

        RAISES NativeUnavailable *MID-ITERATION*, not at call time: the exec
        failure below and the exit-code check in the `finally` both happen
        after the consumer has started pulling. `cli.iter_generate` owns the
        rescue (warn + fall back to torch, or salvage the partial turn) — it is
        the only place that can see both backends. Do not "fix" this by
        swallowing the exception here; a silent empty completion is worse than
        either outcome.
        """
        cmd = self.argv(ids, max_new=max_new, temperature=temperature,
                        top_p=top_p, rep_penalty=rep_penalty,
                        pen_window=pen_window, stop_ids=stop_ids)
        env = dict(os.environ, TRIT_STREAM="1")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True,
                                    bufsize=1, env=env)
        except OSError as exc:
            raise NativeUnavailable(f"cannot execute {self.binary}: {exc}")

        err: list[str] = []
        drain = threading.Thread(
            target=lambda: err.append(proc.stderr.read() or ""), daemon=True)
        drain.start()
        n_out = 0
        self.last_tok_s = None
        try:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("TOK "):
                    n_out += 1
                    yield int(line[4:])
                    if should_stop is not None and should_stop():
                        proc.terminate()
                        return
                elif line.startswith("tok/s:"):
                    m = _TOKS.search(line)
                    if m:
                        self.last_tok_s = float(m.group(1))
        finally:
            try:
                proc.stdout.close()
            except Exception:                               # pragma: no cover
                pass
            rc = proc.wait()
            drain.join(timeout=2)
            # A crash that produced nothing is a backend failure the caller
            # must see; a non-zero code after a terminate() (our own stop) or
            # after tokens were already streamed is not worth blowing up over.
            if rc not in (0, -15) and n_out == 0:
                tail = ("".join(err))[-400:]
                raise NativeUnavailable(
                    f"{self.binary.name} exited {rc}: {tail.strip()}")


class NativeSession:
    """Client for the runner's `--session` stdin/stdout protocol (0.1.7).

    One resident process = one mmap of the container + one KV cache. Each
    request sends the FULL prompt ids; the runner computes the longest common
    prefix against the token history whose KV it already holds and prefills
    only the tail (`ACK reused=<k> new=<n>`), then streams `TOK <id>` lines and
    a final `DONE …` stats line. `C\\n` cancels a decode in flight (poll'd once
    per token); `R\\n` drops the cache; `Q\\n`/EOF exits.

    A binary that predates `--session` rejects the flag and exits before the
    handshake, which surfaces here as SessionError — callers degrade to the
    one-shot path, so old model repos keep working with the new wheel.
    """

    def __init__(self, binary: Path, container: Path, threads: int, ctx: int,
                 runtime_flags: list[str] | None = None):
        self.binary = Path(binary)
        self.container = Path(container)
        self.threads = int(threads)
        self.last_tok_s = None
        self.last_reused = None
        cmd = [str(self.binary), str(self.container), "0", "0",
               str(self.threads), "--session", "--ctx", str(int(ctx))]
        cmd += list(runtime_flags or [])   # 0.1.10: --kv-dtype / --yarn-*
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True,
                                         bufsize=1)
        except OSError as exc:
            raise SessionError(f"cannot execute {self.binary}: {exc}")
        self._err: list[str] = []
        self._drain = threading.Thread(
            target=lambda: self._err.append(self.proc.stderr.read() or ""),
            daemon=True)
        self._drain.start()
        hello = self.proc.stdout.readline().strip()
        if not hello.startswith("SESSION "):
            rc = self.proc.poll()
            self.close()
            raise SessionError(
                f"no session handshake (rc={rc}); the runner binary likely "
                f"predates --session (0.1.7)")
        try:
            self.ctx = int(hello.split("ctx=")[1].split()[0])
        except Exception:
            self.ctx = int(ctx)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def request(self, ids, *, n_new: int, temp: float, topp: float,
                reppen: float, minp: float, penwin: int, seed: int,
                stop_ids, should_stop=None):
        """Yield generated token ids for one request. SessionError before the
        first token (caller may retry one-shot); NativeUnavailable after it
        (partial output already delivered — same contract as one-shot)."""
        stops = ",".join(str(int(s)) for s in stop_ids) if stop_ids else "-"
        line = (f"G {int(n_new)} {temp!r} {topp!r} {reppen!r} {minp!r} "
                f"{int(penwin)} {int(seed)} {stops} "
                f"{','.join(str(int(i)) for i in ids)}\n")
        self.last_tok_s = None
        self.last_reused = None
        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise SessionError(f"session write failed: {exc}")
        n_out = 0
        cancelled = False

        def _cancel():
            try:
                self.proc.stdin.write("C\n")
                self.proc.stdin.flush()
            except (OSError, ValueError):
                pass

        try:
            while True:
                raw = self.proc.stdout.readline()
                if not raw:
                    tail = ("".join(self._err))[-400:]
                    if n_out == 0:
                        raise SessionError(f"session died before first token: "
                                           f"{tail.strip()}")
                    raise NativeUnavailable(
                        f"session runner died mid-decode: {tail.strip()}")
                raw = raw.strip()
                if raw.startswith("TOK "):
                    n_out += 1
                    if not cancelled:
                        yield int(raw[4:])
                        if should_stop is not None and should_stop():
                            cancelled = True   # ask the runner to stop, then
                            _cancel()          # drain quietly to DONE
                elif raw.startswith("ACK"):
                    try:
                        self.last_reused = int(raw.split("reused=")[1].split()[0])
                    except Exception:
                        self.last_reused = None
                elif raw.startswith("DONE"):
                    m = re.search(r"tok_s=([0-9.]+)", raw)
                    if m:
                        self.last_tok_s = float(m.group(1))
                    return
                elif raw.startswith("ERR"):
                    if n_out == 0:
                        raise SessionError(raw)
                    raise NativeUnavailable(f"session error mid-decode: {raw}")
        except GeneratorExit:
            # the consumer abandoned the stream mid-request: cancel and drain
            # to DONE so the protocol stays in sync for the next request.
            _cancel()
            try:
                while True:
                    raw = self.proc.stdout.readline()
                    if not raw or raw.startswith("DONE"):
                        break
            except Exception:                               # pragma: no cover
                pass
            raise

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.write("Q\n")
                    self.proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:                               # pragma: no cover
                pass

    def __del__(self):                                      # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


def eos_ids(tokenizer) -> list[int]:
    """Exactly the ids the torch path stops on, so both paths stop together.

    `cli.iter_generate` passes `eos_token_id=tokenizer.eos_token_id` to
    transformers, which overrides whatever generation_config.json says. Any
    id in this list that is not in that one would make the native path stop
    EARLIER than the reference and break token identity, so this mirrors that
    single source and nothing else.
    """
    src = getattr(tokenizer, "eos_token_id", None)
    if src is None:
        return []
    vals = src if isinstance(src, (list, tuple)) else [src]
    return [int(v) for v in vals if isinstance(v, int)]


def try_load(container: Path, *, device: str, backend: str = "auto",
             draft: bool = False, explicit_bin: str | None = None,
             on_note=None, kv_dtype: str = "fp32",
             yarn_factor: float | None = None,
             yarn_orig_max: int | None = None) -> NativeModel | None:
    """Build a NativeModel, or return None and explain why not.

    `backend="native"` turns every "why not" into a hard error, so a user who
    explicitly asked for the fast path is never silently given the slow one.
    """
    def _no(reason: str):
        if backend == "native":
            raise SystemExit(f"--backend native requested but unavailable: "
                             f"{reason}")
        if on_note:
            on_note(reason)
        return None

    # --kv-dtype int8 and --yarn-factor exist ONLY in the C runtime, so
    # explicitly asking for either hard-requires the native backend (a torch
    # fallback would silently drop them). fp16 (the 0.1.10 DEFAULT) and fp32
    # both degrade gracefully: with no native binary, torch runs its own fp32
    # reference path, which is a superset of correctness, not a silent
    # feature drop -- so they must NOT force native (that would break the
    # torch fallback for every ordinary `fermion chat` on an unsupported
    # platform).
    if kv_dtype == "int8" or yarn_factor is not None:
        feat = "--kv-dtype int8" if kv_dtype == "int8" else "--yarn-factor"
        if backend == "torch":
            raise SystemExit(f"{feat} is a native-runtime feature; it cannot "
                             f"be combined with --backend torch")
        backend = "native"
    if backend == "torch":
        return None
    if device != "cpu":
        return _no(f"--device {device}: the native runtime is a CPU runtime "
                   f"(pass --device cpu, or keep torch for accelerators)")
    binary, why = find_binary(Path(container), explicit=explicit_bin)
    if binary is None:
        return _no(why)
    draft_path = None
    if draft:
        # 0.1.14: the rebuilt runners speak `--draft small.bin` natively
        # (speculative decoding in C, greedy-exact acceptance rule). Gate on
        # the binary actually advertising the flag so a pre-0.1.14 bin/ falls
        # back to the torch spec path exactly as 0.1.13 did.
        if not _binary_supports(binary, "--draft"):
            return _no("this runner binary predates native --draft "
                       "(speculative decoding); delete bin/ and re-download "
                       "the model repo for the rebuilt runner")
        from fermion.cli import container_for, resolve_model
        dkind, dpath = resolve_model(str(draft))
        if dkind == "tiny":
            raise SystemExit("--draft needs a TRTC v4 container "
                             "(the 0.6B), not a tiny QRB8/QRL8 artifact")
        draft_path = Path(container_for(dkind, Path(dpath)))
        _header(draft_path)   # validate the draft container up front
    try:
        return NativeModel(binary, Path(container), note=why,
                           kv_dtype=kv_dtype, yarn_factor=yarn_factor,
                           yarn_orig_max=yarn_orig_max,
                           draft_path=draft_path)
    except NativeUnavailable as exc:
        return _no(str(exc))


def _binary_supports(binary: Path, flag: str) -> bool:
    """Does this runner's usage text advertise `flag`? (rc/streams vary by
    build; usage prints on a no-arg invocation on every shipped runner.)"""
    import subprocess
    try:
        p = subprocess.run([str(binary)], capture_output=True, text=True,
                           timeout=20)
    except Exception:
        return False
    return flag in (p.stdout or "") + (p.stderr or "")


def coverage() -> dict:
    """What this build can and cannot accelerate — for `fermion info`."""
    return {"this_platform": f"{platform.system()}/{platform.machine()}",
            "binary_for_this_platform": platform_tag(),
            "published_binaries": sorted(set(PLATFORM_BINARIES.values()))}


def guard_gguf_for_runner(gguf_path, allow_legacy=False):
    """Gate a .gguf before it is handed to a bundled runner."""
    from fermion._container import check_gguf_provenance
    return check_gguf_provenance(gguf_path, allow_legacy=allow_legacy)
