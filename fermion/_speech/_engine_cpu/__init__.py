"""The Phonon CPU decode engine.

This package is the CPU twin of `fermion/_speech/_engine/` (the MLX runtime):
it loads the same published Phonon-1 artifacts — the identical downloads,
byte for byte — onto a Torch CPU graph whose 196 decoder projections run
through a small closed kernel library shipped as package data
(`fermion/bin/libphonon_cpu-*`, SHA-256 pinned alongside).

Modules:

    artifact.py  — the packed-artifact loader (shared with the container
                   runtime; converts the published storage format into the
                   kernel library's runtime layout, exactly).
    graph.py     — Transformers graph assembly over the official Qwen3-ASR
                   backend (vendored under fermion/_vendor/qwen3_asr_backend,
                   Apache-2.0), with the decoder projections swapped at load.
    runtime.py   — ctypes ownership wrapper for the kernel library.
    model.py     — model construction: FP32 graph, dense prefill, packed
                   decode; loads all three published models (the two
                   hybrid-stored ones through hybrid.py).
    hybrid.py    — exact dequantization of the affine-stored audio tower and
                   embedding the Phonon-1 and Phonon-1 Micro artifacts carry.
    driver.py    — the decode-step driver: one C call per generated token,
                   fused projections, quantized vocabulary head, C attention.
    engine.py    — the `_speech/engine.py`-shaped surface `transcribe`,
                   `listen` and `serve` actually call.

The kernel BINARY is not in this package — it is package data in
`fermion/bin/` (see `gate_cpu.PLATFORM_BINARIES`). Its C sources are not
published.
"""
