"""Vendored, receipt-gated code. Do not edit here.

- hf_ternary/: the TRTC v4 -> Transformers reference integration, vendored
  verbatim from its gated upstream copy (from_pretrained PASS, 0 greedy
  mismatches / 768 tokens, memory honesty 98.93%).
- tiny_ref.py: the float64 reference decoder for the tiny QRB8/QRL8
  artifacts, gated bit-exact against the browser demo (browser/torch/f64
  reference agreement, 12/12 prompts).
"""
