"""Can this machine run Phonon on plain CPU? — the `gate.py` shape, again.

This module is the CPU twin of `fermion/_speech/gate.py` and copies its idiom
deliberately:

  * a dict keyed on `(platform.system(), platform.machine().lower())` decides
    what is published for this machine;
  * runtime deps are probed with `find_spec`, never imported to test, never
    put in `[project] dependencies`;
  * every "why not" becomes a HARD, single-line refusal when the user asked
    for this backend explicitly, and a quiet note otherwise;
  * the kernel binary ships as package data in `fermion/bin/` with a
    `.sha256` pin alongside (the `tv4z-macos-arm64` precedent) and is inert
    on every platform not named below.

`fermion/_speech/backends.py` decides WHEN this gate is consulted: Linux
always, Apple silicon only when the MLX engine is not installed or
when `FERMION_DEVICE=cpu` asks for the CPU engine explicitly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import platform
from pathlib import Path

#: (system, machine) -> packaged kernel binary. Only entries here are ever
#: attempted; the same dict answers "is there a CPU path for this machine".
#: Darwin/arm64 is present because the binary exists and is verified there,
#: but the Mac DEFAULT remains the MLX engine — reaching the CPU engine on a
#: Mac means MLX is absent or `FERMION_DEVICE=cpu` was set.
PLATFORM_BINARIES = {
    ("Linux", "x86_64"): "libphonon_cpu-linux-x86_64.so",
    ("Linux", "aarch64"): "libphonon_cpu-linux-aarch64.so",
    ("Darwin", "arm64"): "libphonon_cpu-macos-arm64.dylib",
    ("Darwin", "aarch64"): "libphonon_cpu-macos-arm64.dylib",
    ("Windows", "amd64"): "phonon_cpu-win-x86_64.dll",
}

#: 0.2.8: platforms where the CPU engine runs Phonon-2 only. Phonon-2's kernel binaries live beside their own loader
#: (`_engine_phonon2_cpu/`, sha256-pinned there); the Phonon-1 CPU library above is not published for these, so a
#: Phonon-1 model is refused in one line (`backends.require_engine_for`) and Phonon-2 runs.
PHONON2_ONLY_PLATFORMS = {
    ("Darwin", "x86_64"),            # Intel Macs
}

#: Probed lazily, never installed by us. torch/transformers/numpy are also
#: base-package deps today; they are probed anyway so this module keeps
#: working if that ever changes, and because probing is free. scipy is here
#: because `read_audio` resamples non-16 kHz files through it.
RUNTIME_MODULES = ("torch", "transformers", "numpy", "safetensors",
                   "soundfile", "scipy")

#: The install lines a refusal prints. On Linux the torch line carries the
#: CPU wheel index so `torch` never resolves to the multi-gigabyte CUDA
#: build; on macOS the plain wheel IS the CPU build. `zstandard` is in the
#: line but not probed, for the same reason as the MLX gate: the model
#: archives are zstd-compressed and a machine with the `zstd` CLI can unpack
#: them without the wheel.
_INSTALL_LINE_LINUX = (
    "pip install --no-deps torch --index-url https://download.pytorch.org/whl/cpu\n"
    "    pip install safetensors soundfile scipy zstandard")
#: On macOS and Windows the plain PyPI torch wheel IS the CPU build, so one
#: line suffices. (On a clean Windows machine torch additionally needs the
#: MSVC runtime — vc_redist.x64.exe — which the README notes; our own
#: kernel library needs only the UCRT that ships with Windows.)
_INSTALL_LINE_MAC = "pip install torch safetensors soundfile scipy zstandard"
_INSTALL_LINE_WINDOWS = _INSTALL_LINE_MAC

#: The env var that selects an engine explicitly: `FERMION_DEVICE=cpu` runs
#: the CPU engine even on a Mac with MLX installed.
DEVICE_ENV = "FERMION_DEVICE"


class CpuSpeechUnavailable(Exception):
    """No usable CPU speech runtime for this platform/environment."""


def requested_device() -> str:
    return os.environ.get(DEVICE_ENV, "").strip().lower()


def _platform_key() -> tuple[str, str]:
    return (platform.system(), platform.machine().lower())


def platform_ok() -> bool:
    return _platform_key() in PLATFORM_BINARIES or _platform_key() in PHONON2_ONLY_PLATFORMS


def phonon2_only() -> bool:
    """True on a platform where the CPU engine serves Phonon-2 but has no Phonon-1 kernel library (Intel Macs)."""
    return _platform_key() in PHONON2_ONLY_PLATFORMS


def install_line() -> str:
    system = platform.system()
    if system == "Darwin":
        return _INSTALL_LINE_MAC
    if system == "Windows":
        return _INSTALL_LINE_WINDOWS
    return _INSTALL_LINE_LINUX


def binary_path(*, verify: bool = True) -> Path:
    """The packaged kernel binary for this machine, sha-checked by default.

    The pin lives alongside the binary (`<name>.sha256`), exactly like
    `fermion/bin/tv4z-macos-arm64.sha256`. A wheel whose binary does not
    match its own pin is a broken install and must be refused, not run.
    """
    key = _platform_key()
    try:
        name = PLATFORM_BINARIES[key]
    except KeyError:
        raise CpuSpeechUnavailable(
            f"no CPU speech binary is published for {key[0]}/{key[1]}")
    path = Path(__file__).resolve().parents[1] / "bin" / name
    if not path.is_file():
        raise CpuSpeechUnavailable(f"packaged binary missing: {path}")
    if verify:
        pin_file = path.with_name(path.name + ".sha256")
        pinned = pin_file.read_text().split()[0]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != pinned:
            raise CpuSpeechUnavailable(
                f"{path.name} does not match its published SHA-256 pin — "
                f"reinstall the package")
    return path


def _missing_modules(names=RUNTIME_MODULES) -> list[str]:
    missing = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


#: Oldest glibc the Linux binaries (manylinux_2_28 builds) and soundfile's bundled libsndfile run on.
GLIBC_FLOOR = (2, 28)


def glibc_version() -> tuple[int, int] | None:
    """(major, minor) of the running glibc on Linux; None elsewhere or when it cannot be read (musl, etc.)."""
    if platform.system() != "Linux":
        return None
    try:
        import ctypes
        f = ctypes.CDLL(None).gnu_get_libc_version
        f.restype = ctypes.c_char_p
        parts = f().decode().split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        try:
            name, ver = platform.libc_ver()
            if name == "glibc" and ver:
                parts = ver.split("."); return int(parts[0]), int(parts[1])
        except Exception:
            pass
    return None


def glibc_below_floor() -> str | None:
    """A one-line note when this Linux is older than the binaries' floor, else None."""
    v = glibc_version()
    if v is None or v >= GLIBC_FLOOR:
        return None
    return (f"this Linux has glibc {v[0]}.{v[1]} and the packed CPU engine binaries need "
            f"{GLIBC_FLOOR[0]}.{GLIBC_FLOOR[1]} or newer (Ubuntu 18.10+/Debian 10+/RHEL 8+); "
            f"using the fp32 reference path instead (same transcripts class, several times slower)")


def audio_io_problem() -> str | None:
    """soundfile must be able to open its libsndfile; on old glibc the wheel's bundled copy does not load."""
    try:
        import soundfile  # noqa: F401
        return None
    except OSError as exc:
        v = glibc_version(); hint = (f" (glibc {v[0]}.{v[1]} here; soundfile's wheel needs {GLIBC_FLOOR[0]}.{GLIBC_FLOOR[1]}+)" if v and v < GLIBC_FLOOR else "")
        musl = v is None and sys.platform.startswith("linux")        # no glibc version readable = musl (Alpine) or an odd libc
        fix = ("    apk add libsndfile   (Alpine / musl: the manylinux wheel bundles libsndfile, the musl wheel does not)"
               if musl else "    sudo apt install libsndfile1   (or the distribution's equivalent)")
        return (f"audio files cannot be read: soundfile could not load libsndfile{hint}. Install the system library and "
                f"retry:\n{fix}\n    [{type(exc).__name__}: {str(exc)[:120]}]")
    except ImportError:
        return None                                  # reported by _missing_modules()


def why_not() -> str | None:
    """None when the CPU engine can run here; otherwise one human sentence."""
    if not platform_ok():
        from fermion._speech import gate as _gate
        if _gate.under_rosetta():
            return _gate.ROSETTA_LINE
        return (f"the CPU speech engine runs on x86-64 and ARM Linux, "
                f"x86-64 Windows, and Apple silicon and Intel Macs — this is "
                f"{platform.system()}/{platform.machine()}.")
    missing = _missing_modules()
    if missing:
        return (f"the CPU speech engine needs {', '.join(missing)}, which "
                f"{'is' if len(missing) == 1 else 'are'} not installed. "
                f"Install with:\n    {install_line()}")
    problem = audio_io_problem()
    if problem:
        return problem
    if glibc_below_floor():
        return None                                  # the fp32 path still runs; the note is carried by describe()
    if phonon2_only():
        return None                                  # Phonon-2's binaries are checked by its own loader; Phonon-1 is refused per model
    try:
        binary_path(verify=True)
    except CpuSpeechUnavailable as exc:
        return str(exc)
    return None


def available() -> bool:
    return why_not() is None


def require(what: str = "CPU speech") -> None:
    """Hard, plain refusal — never a traceback (SystemExit contract)."""
    reason = why_not()
    if reason is not None:
        raise SystemExit(f"{what} is unavailable: {reason}")


def describe() -> dict:
    """For `fermion models` / `/health` — same job as `gate.describe()`."""
    v = glibc_version()
    return {
        "this_platform": f"{platform.system()}/{platform.machine()}",
        "cpu_speech_supported_here": platform_ok(),
        "glibc": (f"{v[0]}.{v[1]}" if v else None), "glibc_floor": f"{GLIBC_FLOOR[0]}.{GLIBC_FLOOR[1]}",
        "packed_engine_note": glibc_below_floor(),
        "supported_platforms": ["Linux/x86_64 (AVX2)",
                                "Linux/aarch64 (NEON)",
                                "Windows/AMD64 (x86-64, AVX2)",
                                "macOS/arm64 (Apple silicon)",
                                "macOS/x86_64 (Intel Macs; Phonon-2)"],
        "runtime_modules": list(RUNTIME_MODULES),
        "missing_modules": _missing_modules() if platform_ok() else None,
        "available": available(),
        "why_not": why_not(),
    }
