"""Drafting (`--draft PATH`): a small Neutrino container proposes tokens, the
big one verifies them.

The draft only ever *proposes*. Every token that reaches the user has been
checked against the big model's own greedy choice, so the text is the text the
big model would have produced alone — drafting changes the speed, not the
output. `fermion verify` re-derives that on the user's machine.

This module does not implement decoding. It loads a second container through
the same banked loader and hands it to the existing generation path as the
assistant model, so all sampling/stopping logic stays where it already is.
"""
from __future__ import annotations

import sys
from pathlib import Path


# `--draft auto` (0.1.14): the canonical draft for the Neutrino family —
# the distilled 0.6B that ships as FermionResearch/Neutrino-0.6B (its card
# carries the acceptance table + the 27,648-token/0-mismatch cert).
CANONICAL_DRAFT = ("FermionResearch/Neutrino-0.6B", "neutrino-0.6b_v4.bin")


def resolve_spec(spec: str) -> str:
    """Map `--draft auto` to a local copy of the canonical draft."""
    if str(spec).strip().lower() != "auto":
        return spec
    from fermion._brand import BRAND
    from huggingface_hub import hf_hub_download
    repo, fname = CANONICAL_DRAFT
    print(f"[{BRAND}] --draft auto: fetching the canonical draft "
          f"{repo}/{fname} (cached after the first run)", file=sys.stderr)
    return hf_hub_download(repo, fname)


def load_draft(spec: str, device: str, dtype: str):
    """Load a draft container. Returns (model, path)."""
    from fermion.cli import load_trtc, resolve_model
    kind, path = resolve_model(spec)
    if kind == "tiny":
        raise SystemExit("--draft needs a TRTC v4 container "
                         "(the 0.6B), not a tiny QRB8/QRL8 artifact")
    # always the torch backend: the draft is consumed by transformers as an
    # `assistant_model`, which needs a real torch graph, not the C runtime.
    model, _tok = load_trtc(kind, path, device, dtype, backend="torch")
    model.eval()
    return model, Path(path)


def attach(base_model, draft_model) -> None:
    """Route every generate() call on `base_model` through the draft.

    Instance-level shadowing of the bound method: the existing generation path
    is untouched and keeps full control of sampling, streaming and stopping.
    """
    original = base_model.generate

    def generate(*a, **kw):
        kw.setdefault("assistant_model", draft_model)
        return original(*a, **kw)

    base_model.generate = generate
    base_model._fermion_draft = draft_model


def enable(base_model, args) -> None:
    """CLI hook: honour `--draft` on chat/generate. Safe to call always."""
    spec = getattr(args, "draft", None)
    if not spec:
        return
    from fermion._brand import BRAND
    if getattr(base_model, "backend", "") == "native":
        # 0.1.14: the native runner drafts in C (--draft passed through by
        # native.try_load; greedy-exact acceptance rule). No torch draft
        # model is loaded at all on this path.
        if getattr(args, "temperature", 0.0) > 0:
            print(f"[{BRAND}] --draft is greedy-only on the native runner; "
                  f"this sampled call runs plain (drop --temperature to "
                  f"draft).", file=sys.stderr)
        else:
            print(f"[{BRAND}] drafting with "
                  f"{Path(str(base_model.draft_path)).name} via the native "
                  f"runner: the draft only proposes; every emitted token is "
                  f"checked against the big model (greedy).", file=sys.stderr)
        return
    draft_model, path = load_draft(spec, args.device, args.dtype)
    attach(base_model, draft_model)
    note = (f"[{BRAND}] drafting with {path.name}: the draft only proposes; "
            "every emitted token is checked against the big model.")
    if getattr(args, "temperature", 0.0) > 0:
        note += (f"\n[{BRAND}] note: the token-identical guarantee is a "
                 "greedy-decoding property (--temperature 0).")
    # Device scoping, measured 2026-07-30 on an L4: under CUDA bf16 the
    # batched verify pass and sequential plain decode can pick different
    # argmax winners at near-ties, so identity does NOT hold there even at
    # temperature 0 (28 divergences / 232 tokens across 8 battery prompts,
    # `fermion verify --device cuda`). CPU/greedy holds. Never print an
    # identity promise the user's own `verify` run would contradict.
    dev = str(getattr(args, "device", "cpu") or "cpu").lower()
    if dev.startswith("cuda"):
        note += (f"\n[{BRAND}] note: on CUDA the activation dtype is bf16 and "
                 "near-tie positions can flip, so the text may differ from a "
                 "plain run — same distribution, not the same tokens. Run "
                 "`fermion verify --device cuda` on your pair to see it. "
                 "The token-identical guarantee is CPU/greedy. Drafting also "
                 "shows no measured speedup on the torch CUDA path (0.99x "
                 "wall on an L4) — plain decode is the better default there.")
    else:
        note += " On CPU/greedy the text is unchanged — only the speed moves."
    print(note, file=sys.stderr)
