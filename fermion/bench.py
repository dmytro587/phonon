"""`fermion bench` — what this model actually does on the machine in front of
you. Single-stream greedy decode, the machine it detected, the artifact size.

Deliberately screenshot-shaped: one block, aligned, no scrollback needed.
"""
from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path

PROMPT = "Explain in a short paragraph why sparse models are cheap to store."


# ------------------------------------------------------------------- machine
def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def machine() -> dict:
    info = {"chip": platform.processor() or platform.machine(),
            "cores": None, "ram_bytes": None,
            "os": f"{platform.system()} {platform.release()}"}
    if platform.system() == "Darwin":
        info["chip"] = _sysctl("machdep.cpu.brand_string") or info["chip"]
        n = _sysctl("hw.ncpu")
        p = _sysctl("hw.perflevel0.logicalcpu")
        e = _sysctl("hw.perflevel1.logicalcpu")
        if n:
            info["cores"] = f"{n} cores"
            if p and e:
                info["cores"] += f" ({p}P + {e}E)"
        mem = _sysctl("hw.memsize")
        info["ram_bytes"] = int(mem) if mem else None
        ver = _sysctl("kern.osproductversion")
        if ver:
            info["os"] = f"macOS {ver}"
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["chip"] = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    info["ram_bytes"] = int(line.split()[1]) * 1024
                    break
        except Exception:
            pass
        import os
        info["cores"] = f"{os.cpu_count()} cores"
    return info


def _gb(b) -> str:
    return f"{b / 2**30:.2f} GiB" if b else "?"


# ----------------------------------------------------------------------- cli
def add_parser(sub, common) -> None:
    ap = sub.add_parser(
        "bench", parents=[common],
        help="measure single-stream decode on this machine (tokens/s, "
             "hardware, artifact size)")
    # bench times a FIXED window named --tokens; the inherited --max-new is
    # meaningless here. Until 0.1.7 it was still advertised in --help and then
    # silently ignored -- hide it, and run() warns anyone who sets it.
    import argparse
    for a in ap._actions:
        if "--max-new" in a.option_strings:
            a.help = argparse.SUPPRESS
    from fermion._tools import add_draft_flag
    add_draft_flag(ap)
    ap.add_argument("--tokens", type=int, default=32,
                    help="tokens to decode per run (default 32)")
    ap.add_argument("--prompt", default=PROMPT, help="benchmark prompt")
    ap.add_argument("--warmup", type=int, default=1,
                    help="warmup tokens before timing (default 1)")


def _decode(model, tokenizer, ids, n, device, assistant=None):
    """A fixed n-token greedy window, timed, on whichever backend is loaded.

    Goes through `fermion.cli.iter_generate` rather than calling
    `model.generate` itself: that is the one decode path, so `bench` times
    the same code `chat`/`generate`/`serve` run — including the native
    runtime — instead of a second, torch-only implementation of decoding.
    `ignore_eos` is what pins the window to exactly n tokens on both
    backends (torch `min_new_tokens`, native: no `--stop-id`).
    """
    from fermion.cli import iter_generate
    res: dict = {}
    t0 = time.time()
    for _piece in iter_generate(model, tokenizer, ids, max_new=n,
                                temperature=0.0, device=device,
                                rep_penalty=1.0, ignore_eos=True,
                                assistant_model=assistant, result=res):
        pass
    return res.get("generated_ids", []), time.time() - t0, res.get("decode_tok_s")


def run(args, kind: str, path: Path) -> int:
    from fermion._brand import BRAND, VERSION
    from fermion.cli import GENERATE_MAX_NEW, _chat_ids, load_trtc
    from fermion.draft import load_draft
    from fermion.inspect import _walk

    if getattr(args, "max_new", GENERATE_MAX_NEW) != GENERATE_MAX_NEW:
        import sys
        print(f"[{BRAND}] bench ignores --max-new; the timed window is "
              f"--tokens (currently {args.tokens})", file=sys.stderr)
    if kind == "tiny":
        raise SystemExit("bench works on TRTC v4 containers")
    path = Path(path)
    import torch

    # `path` is a DIRECTORY whenever the model came from a repo id or a model
    # folder (kind == "trtc-dir"), and `_walk` opens a container file — so
    # `fermion bench --model <dir>` and `fermion bench` with the default HF
    # model both died with `IsADirectoryError` before this line existed. The
    # container, not the folder, is what has a header to walk.
    from fermion.cli import container_for
    container = container_for(kind, path)
    skel = _walk(container, want=lambda li, pk: False)
    params = sum(skel["params"].values())
    size = skel["file_bytes"]

    from fermion.cli import _backend_kwargs
    t0 = time.time()
    model, tokenizer = load_trtc(kind, path, args.device, args.dtype,
                                 **_backend_kwargs(args))
    load_s = time.time() - t0
    draft_model = draft_path = None
    if getattr(args, "draft", None):
        draft_model, draft_path = load_draft(args.draft, args.device,
                                             args.dtype)

    ids = _chat_ids(tokenizer, [{"role": "user", "content": args.prompt}])
    if args.warmup > 0:
        _decode(model, tokenizer, ids, args.warmup, args.device)
    plain, t_plain, dec_plain = _decode(model, tokenizer, ids, args.tokens,
                                        args.device)
    drafted = t_draft = None
    if draft_model is not None:
        drafted, t_draft, _ = _decode(model, tokenizer, ids, args.tokens,
                                      args.device, assistant=draft_model)

    m = machine()
    dtype = "bfloat16" if args.dtype == "auto" else args.dtype
    native = getattr(model, "backend", "torch") == "native"
    w = 72
    out = ["", "  " + "─" * w,
           f"  {BRAND} bench · {container.name}"]
    out.append(f"  {'model':<11}{params / 1e9:.2f}B params · {_gb(size)} on "
               f"disk · one file")
    out.append(f"  {'machine':<11}{m['chip']} · {m['cores'] or '?'} · "
               f"{_gb(m['ram_bytes'])} RAM · {m['os']}")
    # the backend is the headline caveat on any number below it: quote it.
    if native:
        out.append(f"  {'runtime':<11}{BRAND} {VERSION} · native "
                   f"{model.binary.name} · {model.threads} threads · CPU")
    else:
        out.append(f"  {'runtime':<11}{BRAND} {VERSION} · torch "
                   f"{torch.__version__} (reference path) · device "
                   f"{args.device} · {dtype}")
    out.append(f"  {'load':<11}{load_s:.1f}s")
    out.append("  " + "─" * w)
    rate = args.tokens / t_plain
    out.append(f"  {'plain greedy':<20}{args.tokens:>5} tok in "
               f"{t_plain:>7.1f}s  = {rate:>8.2f} tok/s")
    if dec_plain:
        # The native runtime is a one-shot process: this run also paid a
        # process start and a container map/repack, which the line above
        # includes and which the model card's figure does not. Printing only
        # the wall rate makes `bench` contradict our own published number on
        # a big container; printing only the decode rate hides a cost the
        # user really pays. Both, labelled.
        out.append(f"  {'  of which decode':<20}{args.tokens:>5} tok "
                   f"{'':>11} = {dec_plain:>8.2f} tok/s  "
                   f"(runtime steady state; the rest is one-shot "
                   f"start + load)")
    if t_draft:
        rated = args.tokens / t_draft
        out.append(f"  {'with draft':<20}{args.tokens:>5} tok in "
                   f"{t_draft:>7.1f}s  = {rated:>8.2f} tok/s   "
                   f"{t_plain / t_draft:.2f}× faster")
        n = min(len(plain), len(drafted), args.tokens)
        same = plain[:n] == drafted[:n]
        out.append(f"  {'draft artifact':<20}{draft_path.name} "
                   f"({_gb(draft_path.stat().st_size)})")
        out.append("  " + ("output token-identical to plain greedy ✓"
                           if same else
                           "OUTPUT DIFFERS from plain greedy ✗ "
                           "(run `fermion verify`)"))
    out.append("  " + "─" * w)
    # Repro line a skeptic can actually paste: the PyPI distribution is
    # `fermion-research` (BRAND `fermion` is only the command/import name --
    # `pip install fermion` 404s), and the model must be the spec the user
    # gave (repo id or path), not the bare container filename they don't have.
    from fermion._brand import PIP_DIST
    model_spec = getattr(args, "model", None) or str(container)
    out.append(f"  pip install {PIP_DIST} && {BRAND} bench --model {model_spec}")
    out.append("")
    print("\n".join(out))
    return 0
