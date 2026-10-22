"""`fermion speech bench` — what the speech engine does on the machine in front of you.

The speech twin of `fermion bench`: one aligned block, screenshot-shaped. It loads the
speech model exactly as `transcribe` does (same catalog resolution, same engine
dispatch, same decode call), decodes the audio files you name, and prints the
machine, the engine (MLX or CPU), the model, the speed levers in force, load time,
milliseconds of compute per second of audio, × realtime, and peak memory.

    fermion speech bench --audio talk.wav                 # default model
    fermion speech bench --model phonon-2 --audio a.wav --audio b.flac --runs 3

Registered EARLY (see `_tools.EARLY`) for the same reason `transcribe` is: a speech
model is not a TRTC container and must never reach `resolve_model`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

from fermion._brand import BRAND, SPEECH_MODEL_ID, DEFAULT_SPEECH_MODEL_ID, VERSION


def _console_safe(text: str) -> str:
    """Windows cp1252 consoles cannot print the rule (─) or the middle dot (·): fall back to ASCII there (0.2.2)."""
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
    try:
        text.encode(enc)
        return text
    except (UnicodeEncodeError, LookupError):
        return text.replace("─", "-").replace("·", "|").encode(enc, errors="replace").decode(enc, errors="replace")


def _peak_rss_gb() -> float | None:
    """Peak resident set in GB; `resource` is Unix-only, Windows reads it from psutil-free Win32 (0.2.1: the module-level
    import made every `fermion` command fail on Windows with ModuleNotFoundError)."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1e9 if platform.system() == "Darwin" else 1e6)
    except ImportError:
        pass
    try:
        import ctypes, ctypes.wintypes as wt
        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / 1e9
    except Exception:
        pass
    return None


def add_parser(sub, common) -> None:
    """`common` is accepted and unused, exactly as in `transcribe.add_parser`."""
    p = sub.add_parser("speech", help="speech-model tools: `speech bench`")
    ss = p.add_subparsers(dest="speech_cmd", required=True)
    b = ss.add_parser("bench", help="measure the speech engine on this machine "
                                   "(ms of compute per second of audio, × realtime)")
    b.add_argument("model_pos", nargs="?", metavar="MODEL",
                   help="the speech model to measure: phonon-2, phonon-1, phonon-1-big, phonon-1-micro; "
                        "`fermion speech bench phonon-2 --audio clip.wav`")
    b.add_argument("--model", default=None, help=argparse.SUPPRESS)
    b.add_argument("--audio", action="append", default=None, metavar="FILE",
                   help="audio file to decode (repeatable; wav/flac/ogg/aiff, any length)")
    b.add_argument("--runs", type=int, default=2, help="timed passes over the files after one warm-up (default 2)")
    b.add_argument("--json", action="store_true", help="emit the table as one JSON object")


def _machine() -> str:
    from fermion.bench import machine, _gb
    m = machine()
    return f"{m['chip']} · {m['cores'] or '?'} · {_gb(m['ram_bytes'])} RAM · {m['os']}"


def run_early(args) -> int:
    from fermion import _modelarg
    _modelarg.take_model(args, "speech bench", "fermion speech bench phonon-2 --audio clip.wav")
    if getattr(args, "speech_cmd", None) != "bench":
        raise SystemExit("usage: fermion speech bench --audio FILE [--model M] [--runs N]")
    if not args.audio:
        raise SystemExit(f"{BRAND} speech bench needs at least one --audio FILE (any wav/flac you have; "
                         f"10–60 s of speech gives a steady number)")
    files = [Path(a) for a in args.audio]
    for f in files:
        if not f.is_file():
            raise SystemExit(f"{f}: not a file")
    from fermion._speech import backends, fetch
    from fermion.transcribe import _resolve
    repo, key, pin, local_dir = _resolve(args.model)
    engine_kind = backends.resolve("`fermion speech bench`")
    backends.require_engine_for(engine_kind, pin.get("backend", ""))
    model_dir = local_dir if local_dir is not None else fetch.ensure(repo, key, pin)
    t0 = time.perf_counter()
    speech = backends.load(engine_kind, model_dir, profile=key, backend=pin["backend"], quiet=True)
    load_s = time.perf_counter() - t0
    desc = speech.describe()
    # warm-up (Metal shader compile on the first call), then timed passes
    for f in files:
        speech.transcribe(f)
    rows = []
    for _ in range(max(1, args.runs)):
        for f in files:
            text, dec_s, audio_s = speech.transcribe(f)
            rows.append({"file": f.name, "audio_s": round(audio_s, 2), "decode_s": round(dec_s, 4),
                         "ms_per_audio_s": round(1000 * dec_s / max(audio_s, 1e-9), 2), "words": len(text.split())})
    ta = sum(r["audio_s"] for r in rows); td = sum(r["decode_s"] for r in rows)
    rss_gb = _peak_rss_gb()
    peak_mlx = None
    if engine_kind == "mlx":
        try:
            import mlx.core as mx
            peak_mlx = mx.get_peak_memory() / 1e9
        except Exception:
            pass
    levers = desc.get("levers")
    out = {"brand": BRAND, "version": VERSION, "model": desc.get("profile") or key, "model_dir": str(model_dir),
           "engine": engine_kind, "runtime": desc.get("runtime"), "levers": levers, "levers_error": desc.get("levers_error"),
           "machine": _machine(), "load_s": round(load_s, 2), "files": len(files), "runs": args.runs,
           "audio_s": round(ta, 1), "decode_s": round(td, 3), "ms_per_audio_s": round(1000 * td / max(ta, 1e-9), 2),
           "x_realtime": round(ta / max(td, 1e-9), 1), "peak_rss_gb": (round(rss_gb, 2) if rss_gb is not None else None),
           "peak_mlx_gb": None if peak_mlx is None else round(peak_mlx, 2), "rows": rows}
    if args.json:
        print(json.dumps(out, indent=1)); return 0
    w = 72
    lines = ["", "  " + "─" * w, f"  {BRAND} speech bench · {out['model']}",
             f"  {'machine':<11}{out['machine']}",
             f"  {'engine':<11}{engine_kind} · {out['runtime'] or ''}",
             f"  {'levers':<11}{', '.join(levers) if levers else 'none'}" + (f"  (lever error: {out['levers_error']})" if out["levers_error"] else ""),
             f"  {'load':<11}{load_s:.2f}s",
             "  " + "─" * w,
             f"  {'audio':<11}{len(files)} file(s) × {args.runs} run(s) = {ta:.1f} s of audio",
             f"  {'compute':<11}{td:.3f} s  = {out['ms_per_audio_s']:.2f} ms per second of audio  = {out['x_realtime']:.1f}× realtime",
             f"  {'memory':<11}peak RSS " + (f"{rss_gb:.2f} GB" if rss_gb is not None else "n/a") + (f" · MLX peak {peak_mlx:.2f} GB" if peak_mlx is not None else ""),
             "  " + "─" * w,
             f"  pip install fermion-research && {BRAND} speech bench {args.model} --audio {files[0]}", ""]
    print(_console_safe("\n".join(lines)))
    return 0


def run(args, kind, path) -> int:                           # pragma: no cover
    return run_early(args)
