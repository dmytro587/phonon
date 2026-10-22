"""Can this machine run Phonon? — the `native.try_load()` shape, for speech.

The idiom is copied deliberately from `fermion/native.py`:

  * a dict keyed on `(platform.system(), platform.machine().lower())` decides
    what is published for this machine (`native.PLATFORM_BINARIES`);
  * `try_load()` returns `None` PLUS a human reason when the fast path is
    unavailable, and turns every "why not" into a HARD ERROR when the user
    asked for that backend explicitly.

The explicit ask, for speech, is "the user typed a speech verb". `fermion
transcribe` on Linux is not a fallback situation — there is nothing to fall
back TO — so it is always a hard, plain, single-line refusal. What must NEVER
happen is a traceback: a Linux user typing `fermion transcribe` gets an
explanation, and a Linux user typing `fermion chat` must not even be able to
tell this module exists (nothing here is imported from `cli.py`'s top level,
and `chat` never reaches it).
"""
from __future__ import annotations

import importlib.util
import platform

# Where the MLX engine runs. Same shape as native.PLATFORM_BINARIES, same
# rule: only the entries here are ever attempted.
#
# Apple silicon only, because MLX is Apple-silicon-only by construction: the
# decoder's five-value weights are decoded into MLX's native 2-bit quantized
# matmuls (`mx.quantized_matmul`, `nn.QuantizedLinear`,
# `nn.QuantizedEmbedding`). This gate covers the MLX engine only — on other
# supported machines `_speech/backends.py` dispatches to the CPU engine
# (`gate_cpu.py`), which serves the same models.
SUPPORTED_PLATFORMS = {
    ("Darwin", "arm64"),
    ("Darwin", "aarch64"),
}

# Imported lazily, never installed by us, and deliberately NOT in
# `[project] dependencies`: Neutrino users on Linux, Windows and Intel Macs
# must be unaffected by the existence of speech, and `pip install
# fermion-research` must stay one line with no extras syntax.
RUNTIME_MODULES = ("mlx", "mlx_audio", "mlx_lm", "soundfile", "scipy")

#: The single line a user is told to run. One pip command, no extras syntax.
#:
#: `zstandard` is in the line but NOT in `RUNTIME_MODULES` above, on purpose.
#: The model archives are zstd-compressed, and the only decompressors are the
#: `zstandard` wheel, Python 3.14's `compression.zstd`, and the `zstd`
#: command-line tool. macOS ships NONE of them: there is no `/usr/bin/zstd`,
#: so a Mac without Homebrew has no way to unpack what it just downloaded.
#: Installing it up front turns that into a non-event. It is not a probe
#: requirement because a machine that already has the command-line tool is
#: perfectly able to run speech without the wheel, and refusing there would be
#: a false negative.
INSTALL_LINE = "pip install mlx mlx-audio mlx-lm soundfile scipy zstandard"

#: What `fermion listen` needs ON TOP of the speech runtime: microphone
#: capture. Deliberately NOT in `RUNTIME_MODULES`: `transcribe` and `serve`
#: never touch a microphone, and they must not acquire a requirement because a
#: sibling verb grew one — this is probed by `require_listen()` alone, which
#: only `listen` calls. `INSTALL_LINE` above does not change either:
#: `mlx-audio` already depends on `sounddevice>=0.5.3`, so the documented line
#: installs it transitively; the probe exists for the environment where it was
#: removed or broken, which must be a one-line refusal, not a traceback.
LISTEN_MODULES = ("sounddevice",)


#: What a Rosetta-translated Python is told. The install line is useless to
#: it (the Apple-silicon wheels do not resolve for an x86-64 interpreter),
#: so the only fix named is the real one.
ROSETTA_LINE = (
    "this Python is running under Rosetta (an Intel build on an Apple-silicon "
    "Mac); install an arm64 Python to use the GPU/CPU engines. "
    "`python3 -c 'import platform; print(platform.machine())'` should print "
    "arm64. (FERMION_DEVICE=cpu runs the Intel-Mac CPU engine under Rosetta; "
    "it is much slower than the Apple-silicon engines.)")


class SpeechUnavailable(Exception):
    """No usable speech runtime for this platform/environment."""


def platform_ok() -> bool:
    return (platform.system(), platform.machine().lower()) in SUPPORTED_PLATFORMS


def under_rosetta() -> bool:
    """Is this an x86-64 Python translated by Rosetta on an Apple-silicon Mac?

    `platform.machine()` says `x86_64` for such a process, which both gates
    would otherwise report as an Intel Mac and refuse with an install line
    that cannot work. `sysctl.proc_translated` is 1 for a translated
    process, 0 for a native one, and absent on Intel hardware.
    """
    if platform.system() != "Darwin" or platform.machine().lower() != "x86_64":
        return False
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() == "1"
    except Exception:                                        # noqa: BLE001
        return False


def _missing_modules(names=RUNTIME_MODULES) -> list[str]:
    """Which of `names` are absent — WITHOUT importing any of them.

    `find_spec` only consults the import machinery's finders. Importing `mlx`
    to test for it would load a multi-megabyte Metal runtime just to print an
    error message, and would do it inside `fermion models`, which is supposed
    to be instant.
    """
    missing = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def why_not() -> str | None:
    """None when speech can run here; otherwise one human sentence."""
    if not platform_ok():
        if under_rosetta():
            return ROSETTA_LINE
        return (f"Phonon's MLX engine runs on Apple silicon (arm64 macOS) "
                f"only — this is {platform.system()}/{platform.machine()}. "
                f"On x86-64 Windows, on Linux (x86-64 and ARM) and on Intel "
                f"Macs the CPU engine serves the same models "
                f"(`fermion models` shows what this machine can run). The "
                f"language models (`fermion chat`, `fermion generate`, "
                f"`fermion serve`) run here normally.")
    missing = _missing_modules()
    if missing:
        return (f"the speech runtime needs {', '.join(missing)}, which "
                f"{'is' if len(missing) == 1 else 'are'} not installed. These "
                f"are Apple-silicon-only wheels, so they are not dependencies "
                f"of `fermion-research` itself — Linux and Intel users must "
                f"not be forced to resolve them. Install them with:\n"
                f"    {INSTALL_LINE}")
    return None


def available() -> bool:
    return why_not() is None


def require(what: str = "speech") -> None:
    """Hard, plain refusal — never a traceback. The explicit-ask path.

    `native.try_load()` raises SystemExit for `--backend native requested but
    unavailable`; this is the same contract, and SystemExit is what makes it a
    clean message-and-exit-1 rather than a stack trace.
    """
    reason = why_not()
    if reason is not None:
        raise SystemExit(f"{what} is unavailable: {reason}")


def why_not_listen() -> str | None:
    """`why_not()`, plus microphone capture. Consulted ONLY by `listen`.

    A LAZY, PER-COMMAND GATE: this is deliberately not folded into
    `why_not()`, so `transcribe`, `serve` and `fermion models` keep exactly
    the requirements they had — a machine that can transcribe but has a broken
    `sounddevice` must keep transcribing, and be refused only the verb that
    actually opens a microphone.
    """
    reason = why_not()
    if reason is not None:
        return reason
    missing = _missing_modules(LISTEN_MODULES)
    if missing:
        return (f"live microphone capture needs {', '.join(missing)}, which "
                f"{'is' if len(missing) == 1 else 'are'} not installed. The "
                f"speech install line provides it (via mlx-audio):\n"
                f"    {INSTALL_LINE}")
    return None


def require_listen(what: str = "`fermion listen`") -> None:
    """`require()`, for the microphone path. Same contract: SystemExit, one
    plain line, never a traceback."""
    reason = why_not_listen()
    if reason is not None:
        raise SystemExit(f"{what} is unavailable: {reason}")


def try_load(*, on_note=None, explicit: bool = True):
    """Import the engine, or return None and explain why not.

    Mirrors `native.try_load(container, backend=...)`: `explicit=True` (a user
    who typed a speech verb) turns every "why not" into a hard error;
    `explicit=False` (an internal probe, e.g. `fermion models` marking what is
    runnable) returns None and optionally notes the reason.
    """
    reason = why_not()
    if reason is None:
        from fermion._speech import engine
        return engine
    if explicit:
        raise SystemExit(f"speech is unavailable: {reason}")
    if on_note:
        on_note(reason)
    return None


def describe() -> dict:
    """What this build can and cannot run — for `fermion models` / `/health`.

    Same job as `native.coverage()`.
    """
    return {
        "this_platform": f"{platform.system()}/{platform.machine()}",
        "speech_supported_here": platform_ok(),
        "supported_platforms": ["macOS/arm64 (Apple silicon)"],
        "runtime_modules": list(RUNTIME_MODULES),
        "missing_modules": _missing_modules() if platform_ok() else None,
        "available": available(),
        "why_not": why_not(),
    }
