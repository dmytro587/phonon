"""Which speech engine runs here — the one dispatch seam.

Phonon has two engines behind one surface:

  * the MLX engine (`_speech/engine.py`) — the Apple-silicon default, the
    configuration the Mac app ships;
  * the CPU engine (`_speech/_engine_cpu/engine.py`) — plain-CPU decoding
    through a small packed kernel library shipped as package data; the path
    on Linux (x86-64 and 64-bit ARM), and on a Mac without the MLX
    runtime.

The rules, in order:

  1. `FERMION_DEVICE=cpu` selects the CPU engine explicitly, anywhere it is
     published. `FERMION_DEVICE=mlx` selects the MLX engine explicitly.
  2. Apple silicon defaults to MLX. When the MLX runtime is not installed
     but the CPU engine can run, the CPU engine is used with one stderr
     note naming both the choice and the way back.
  3. Linux (x86-64 and 64-bit ARM), Windows x86-64 and Intel Macs (0.2.8;
     Phonon-2 only) run the CPU engine.
  4. Anything else is a one-line refusal naming what is supported.

Both engines load the SAME published artifacts — identical downloads, byte
for byte — and both decode deterministically. Transcripts are same-accuracy
across engines, not word-identical.
"""
from __future__ import annotations

import os
import sys

from fermion._brand import (BRAND, SPEECH_BIG_MODEL_NAME,
                            SPEECH_MICRO_MODEL_NAME, SPEECH_MODEL_NAME)
from fermion._speech import gate, gate_cpu


def resolve(what: str) -> str:
    """'mlx' or 'cpu' for this machine/request; SystemExit refusal otherwise.

    `what` names the user's command for the refusal line, exactly like
    `gate.require()`.
    """
    # A Rosetta-translated Python is told so FIRST, before either gate can
    # call it an Intel Mac: the fix is a different Python, not a different
    # install line. (Both gates also carry this line, for `fermion models`.)
    device = gate_cpu.requested_device()
    if gate.under_rosetta() and device != "cpu":
        raise SystemExit(f"{what} is unavailable: {gate.ROSETTA_LINE}")
    if device == "cpu":
        gate_cpu.require(what)
        _glibc_note_once()
        return "cpu"
    if device in ("mlx", "metal", "gpu"):
        gate.require(what)
        return "mlx"
    if device:
        raise SystemExit(
            f"{gate_cpu.DEVICE_ENV}={device!r} does not name a speech engine "
            f"— use `cpu`, `mlx`, or unset it for the platform default.")
    if gate.platform_ok():
        if gate.available():
            return "mlx"
        if gate_cpu.available():
            print(f"[{BRAND}] the MLX runtime is not installed; using the "
                  f"CPU engine (slower). `{gate.INSTALL_LINE}` restores the "
                  f"Apple-silicon default; FERMION_DEVICE=cpu selects the "
                  f"CPU engine explicitly.", file=sys.stderr)
            _glibc_note_once()
            return "cpu"
        raise SystemExit(
            f"{what} is unavailable: {gate.why_not()}\n"
            f"    (the CPU engine can also run here once its runtime is "
            f"installed: {gate_cpu.install_line()})")
    gate_cpu.require(what)
    _glibc_note_once()
    return "cpu"


#: Catalog profile key -> the name a person reads. The banner below must
#: never print an internal profile key.
_HUMAN_NAMES = {"audio6": SPEECH_MODEL_NAME, "parity": SPEECH_BIG_MODEL_NAME,
                "micro": SPEECH_MICRO_MODEL_NAME}



def _glibc_note_once() -> None:
    """Below the glibc floor the CPU engine takes the fp32 path: say so HERE, before the model download, once per process."""
    note = gate_cpu.glibc_below_floor()
    if note and not os.environ.get("FERMION_P2_GLIBC_NOTED"):
        print(f"[{BRAND}] {note}", file=sys.stderr)
        os.environ["FERMION_P2_GLIBC_NOTED"] = "1"

def _first_run_marker(engine_kind: str):
    """A marker file that exists once THIS environment has loaded an engine.

    The ~20 s first-invocation cost (bytecode compile of the dependency
    stack plus the Metal kernel compile) is paid once per Python
    environment, not once per machine or per model, so the marker is keyed
    on `sys.prefix`. It lives in the CLI's own cache next to the models;
    when that is unwritable the banner simply keeps its short form.
    """
    import hashlib
    try:
        from fermion._speech import fetch
        key = hashlib.sha1(sys.prefix.encode()).hexdigest()[:12]
        return fetch.speech_root() / f".ran-{engine_kind}-{key}"
    except Exception:                                        # noqa: BLE001
        return None


def banner(engine_kind: str, profile: str) -> None:
    """The one always-on stderr line at engine start.

    Names the engine and the model so a benchmark or a bug report can never
    be silently mis-attributed (the CPU engine used to be named only when
    MLX was missing or under --verbose), and says once per environment that
    the first run compiles and later runs are fast, so a first-run wall
    clock is not mistaken for the steady state.
    """
    marker = _first_run_marker(engine_kind)
    first = marker is None or not marker.exists()
    name = _HUMAN_NAMES.get(profile, "local model")
    note = ""
    if first:
        note = (" · first run in a new environment compiles shaders (~20 s), "
                "later runs are fast" if engine_kind == "mlx" else
                " · first run in a new environment compiles the runtime, "
                "later runs are faster")
    line = f"[{BRAND}] decode backend: {engine_kind} · {name}{note}"
    enc = getattr(sys.stderr, "encoding", None) or "utf-8"
    try:
        line.encode(enc)
    except (UnicodeEncodeError, LookupError):        # cp1252/cp437 consoles (Windows) cannot show the middle dot
        line = line.replace("·", "|").replace("~", "~")
    print(line, file=sys.stderr)


def _mark_ran(engine_kind: str) -> None:
    marker = _first_run_marker(engine_kind)
    if marker is not None and not marker.exists():
        try:
            marker.touch()
        except OSError:
            pass


def require_engine_for(engine_kind: str, backend: str) -> None:
    """Refuse an engine/model pair that cannot run, BEFORE any download.

    Phonon-2 (backend `phonon2-five-value`) runs on Apple silicon through MLX in this release; the CPU
    engine does not load it. Called by every entry point right after `resolve()`, with the pin's backend.
    """
    # 2026-09-28: Phonon-2 now has a CPU engine too (`engine_phonon2_cpu.py`: the container expanded to dense fp32
    # in the stock transformers graph, greedy TDT loop; ~13x realtime on 12 cores, 2.5 GB resident). Nothing to refuse.
    # 0.2.8: Intel Macs run Phonon-2 on the CPU engine; the Phonon-1 kernel library is not published there.
    if engine_kind == "cpu" and backend != "phonon2-five-value" and gate_cpu.phonon2_only():
        import platform
        raise SystemExit(f"Phonon-1 is not available on {platform.system()}/{platform.machine()} (Intel Mac): its CPU "
                         f"kernel library is not published for this platform. Phonon-2 runs here: "
                         f"`fermion transcribe phonon-2 recording.wav`.")
    return None


def load(engine_kind: str, model_dir, *, profile: str, backend: str,
         quiet: bool = True):
    """Load one engine over an unpacked model directory.

    Prints the always-on `banner()` first (every entry point loads through
    here), then the engine's own verbose load line if `quiet` is False.
    """
    require_engine_for(engine_kind, backend)
    banner(engine_kind, profile)
    # The engines import torch / transformers / MLX on first use; the
    # environment shields must be in place before that (lazy since 0.1.25).
    from fermion import _compat
    _compat.prepare_torch_stack()
    if engine_kind == "mlx":
        from fermion._speech import engine as _mlx
        speech = _mlx.load(model_dir, profile=profile, backend=backend,
                           quiet=quiet)
    elif backend == "phonon2-five-value":
        from fermion._speech import engine_phonon2_cpu as _p2cpu
        speech = _p2cpu.load(model_dir, profile=profile, backend=backend, quiet=quiet)
    else:
        from fermion._speech._engine_cpu import engine as _cpu
        speech = _cpu.load(model_dir, profile=profile, quiet=quiet)
    _mark_ran(engine_kind)
    return speech


def describe() -> dict:
    """Merged engine coverage for `fermion models` / `/health`."""
    mlx = gate.describe()
    cpu = gate_cpu.describe()
    available = bool(mlx["available"] or cpu["available"])
    if available:
        why = None
    elif mlx["speech_supported_here"]:
        why = mlx["why_not"]
    else:
        why = cpu["why_not"] or mlx["why_not"]
    return {
        "this_platform": mlx["this_platform"],
        "speech_supported_here": bool(mlx["speech_supported_here"]
                                      or cpu["cpu_speech_supported_here"]),
        "supported_platforms": ["macOS/arm64 (Apple silicon; MLX engine)",
                                "Linux/x86_64 (AVX2; CPU engine)",
                                "Linux/aarch64 (NEON; CPU engine)",
                                "Windows/AMD64 (x86-64, AVX2; CPU engine)",
                                "macOS/x86_64 (Intel Macs; CPU engine, Phonon-2)"],
        "available": available,
        "why_not": why,
        "engine": ("mlx" if mlx["available"]
                   else "cpu" if cpu["available"] else None),
        "note": (None if mlx["available"] else cpu.get("packed_engine_note")),
        "engines": {"mlx": mlx, "cpu": cpu},
    }
