"""`fermion verify` — run our release gate on the user's own machine.

Decodes a fixed prompt battery twice, greedily: once with the big model alone,
once with the small model drafting for it. Then compares the two token streams
position by position. Drafting is only worth trusting if those streams are the
same stream, so we let the user prove it locally instead of asking them to
believe a receipt from our lab. Exits non-zero on any divergence.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

BATTERY = [
    "What is the capital of France?",
    "List the first ten prime numbers.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Explain what rotary position embeddings do, in two sentences.",
    "Count from 1 to 20.",
    "Who wrote Pride and Prejudice?",
    "Summarise in three sentences why the sky is blue.",
    "Translate 'good morning' into French, Spanish and German.",
    "Name three planets and one fact about each.",
    "Write a haiku about a lighthouse.",
    "Give me a shell one-liner that counts the lines in every .py file.",
    "What are the trade-offs between a B-tree and a hash index?",
]


def _short(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


def _greedy(model, tokenizer, ids, max_new: int, device: str, assistant=None):
    import torch
    inp = torch.tensor([ids], device=device)
    kw = dict(input_ids=inp,
              attention_mask=torch.ones_like(inp),
              max_new_tokens=max_new, do_sample=False,
              eos_token_id=tokenizer.eos_token_id,
              pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if assistant is not None:
        kw["assistant_model"] = assistant
    t0 = time.time()
    out = model.generate(**kw)
    return out[0].tolist()[len(ids):], time.time() - t0


def add_parser(sub, common) -> None:
    ap = sub.add_parser(
        "verify", parents=[common],
        help="prove on your own machine that drafted output is "
             "token-identical to plain greedy output")
    from fermion._tools import add_draft_flag
    add_draft_flag(ap)
    ap.add_argument("--prompts", type=int, default=8,
                    help="how many battery prompts to run "
                         f"(default 8, max {len(BATTERY)})")
    ap.set_defaults(max_new=32)   # battery default: 8 x 32 = 256 tokens


def run(args, kind: str, path: Path) -> int:
    from fermion._brand import BRAND, VERSION
    from fermion.cli import _chat_ids, load_trtc
    from fermion.draft import load_draft
    from fermion.inspect import sha256_file

    if kind == "tiny":
        raise SystemExit("verify works on TRTC v4 containers")
    if not getattr(args, "draft", None):
        raise SystemExit(
            "verify needs both models: --model <8B container> "
            "--draft <0.6B container>. The whole point is to check the pair.")
    n = max(1, min(args.prompts, len(BATTERY)))
    prompts = BATTERY[:n]

    # 0.1.14: verify PROVES the identity claim, and the claim's regime is
    # fp32/greedy — so verify runs fp32 unless the user overrides. Under
    # `auto` (bfloat16) the batched verify forward and the sequential plain
    # decode can pick different argmax winners at near-ties even on CPU
    # (measured 2026-07-31 on Linux x86: 4/192 tail flips at bf16, 0/192 at
    # fp32), which would fail the proof for a reason that is dtype physics,
    # not a broken pair.
    if args.dtype == "auto":
        args.dtype = "float32"
        print(f"[{BRAND}] verify runs --dtype float32 (the identity "
              f"guarantee's regime); pass --dtype bfloat16 to probe the "
              f"chat-default dtype, where near-tie flips are expected.",
              file=sys.stderr)

    # both sides torch: this gate is about speculative decoding, which is a
    # torch-graph feature (`assistant_model`). The native runtime has no
    # drafting mode, so forcing it here would compare unlike things.
    if str(args.device).lower().startswith("cuda") and args.dtype == "float32":
        # fp32 must mean fp32: on Ampere/Ada, TF32 can silently stand in for
        # fp32 GEMMs and reintroduce the exact near-tie flips --dtype float32
        # exists to remove. Pin both switches before any model loads.
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, tokenizer = load_trtc(kind, path, args.device, args.dtype,
                                 backend="torch")
    draft_model, draft_path = load_draft(args.draft, args.device, args.dtype)
    base_path = Path(path)

    print(f"{BRAND} verify · release gate, run locally · {BRAND} {VERSION}")
    for label, p in (("brain", base_path), ("draft", draft_path)):
        if p.is_file():
            print(f"  {label}   {p.name}  {p.stat().st_size / 2**30:.2f} GiB"
                  f"  sha256 {sha256_file(p)[:16]}…")
    print(f"  device  {args.device} · dtype {args.dtype} · greedy · "
          f"{n} prompts × {args.max_new} tokens")
    print()
    print(f"  {'#':>2}  {'prompt':<44}{'tok':>5}{'plain':>9}"
          f"{'drafted':>10}{'result':>9}")

    compared = 0
    divergences = []
    t_plain = t_draft = 0.0
    for i, prompt in enumerate(prompts, 1):
        ids = _chat_ids(tokenizer, [{"role": "user", "content": prompt}])
        a, ta = _greedy(model, tokenizer, ids, args.max_new, args.device)
        b, tb = _greedy(model, tokenizer, ids, args.max_new, args.device,
                        assistant=draft_model)
        t_plain += ta
        t_draft += tb
        # assisted generation may hand back one block past the cap; the gate is
        # over the requested window, not over the overshoot.
        a, b = a[:args.max_new], b[:args.max_new]
        k = min(len(a), len(b))
        bad = [j for j in range(k) if a[j] != b[j]]
        if len(a) != len(b):
            bad.append(k)
        compared += k
        for j in bad:
            divergences.append((i, prompt, j,
                                a[j] if j < len(a) else None,
                                b[j] if j < len(b) else None))
        mark = "OK" if not bad else f"{len(bad)} BAD"
        print(f"  {i:>2}  {_short(prompt, 43):<44}{k:>5}{ta:>8.1f}s"
              f"{tb:>9.1f}s{mark:>9}", flush=True)

    print("  " + "-" * 78)
    speed = f" · drafted {t_plain / t_draft:.2f}× wall-clock" if t_draft else ""
    verdict = "PASS" if not divergences else "FAIL"
    print(f"  {compared} tokens compared · {len(divergences)} divergences"
          f" · {verdict}{speed}")
    if divergences:
        print("\n  divergences:")
        for i, prompt, j, ta_, tb_ in divergences[:20]:
            da = tokenizer.decode([ta_]) if ta_ is not None else "<end>"
            db = tokenizer.decode([tb_]) if tb_ is not None else "<end>"
            print(f"   prompt {i} position {j}: plain {ta_} {da!r} "
                  f"vs drafted {tb_} {db!r}")
        print("\n  Drafted output is NOT token-identical here. Do not ship "
              "this pair.", file=sys.stderr)
        if args.dtype in ("auto", "bfloat16", "float16"):
            print("\n  Scope: this run compared reduced-precision legs "
                  f"(dtype {args.dtype}). Divergences\n  here are consistent "
                  "with the documented near-tie class — the batched verify\n"
                  "  forward and the sequential plain decode can pick "
                  "different argmax winners\n  where the top two logits "
                  "nearly tie (measured on CUDA bf16 AND Linux-CPU\n  bf16); "
                  "inspect the positions listed above to confirm. The "
                  "token-identical\n  guarantee is fp32/greedy. Re-run with "
                  "--dtype float32 to gate the fp32 pair.",
                  file=sys.stderr)
        raise SystemExit(1)
    print("\n  Every token the drafted run emitted is the token the big model "
          "would have\n  emitted on its own. Drafting moved the clock, not "
          "the text.")
    return 0
