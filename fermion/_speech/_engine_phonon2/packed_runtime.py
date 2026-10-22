"""Phonon-2 packed MLX runtime: five-value Parakeet-TDT over mlx-audio's graph.

The container (``fermion-five-value-parakeet-v1``) stores every quantised
encoder linear as a trit plane (5 trits/byte) + one residual bit per nonzero +
per-row fp16 ``lo``/``hi``.  Each such row takes values in {0, +-lo, +-hi}.

At load we expand each record into TWO native MLX 2-bit affine planes and run
two ``mx.quantized_matmul`` calls (exactly the mechanism ``v18_runtime.py``
uses for Phonon-1's decoder), so no new kernel is involved:

    plane A (base)     : q_a = sign + 1 in {0,1,2}        w_a = lo * q_a - lo   = sign * lo
    plane B (residual) : q_b = 1 + sign * is_hi in {0,1,2} w_b = d  * q_b - d    = sign * is_hi * d
    d = hi - lo   ->   w_a + w_b = sign * (lo + is_hi * d) = {0, +-lo, +-hi}

``lo`` and ``d`` are per-row; MLX wants per-group (128) metadata, so the row
value is broadcast across the ``in_features // 128`` groups at load
(``materialize_runtime_metadata`` in v18_runtime does the same).  With fp32
metadata the expansion is exact in IEEE arithmetic: lo is an fp16 value, d is
the fp32 difference of two fp16 values (exact), and lo + d == hi exactly.

Nothing here reads the container or maps HF names to mlx-audio names for the
dense tensors -- those live in the shared decoder written once by the Detta C3
lane (``fermion_container.py`` / the name map), imported below.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

GROUP = 128


# --------------------------------------------------------------------------- names
def five_value_hf_to_mlx(hf_name: str) -> str:
    """HF ``ParakeetForTDT`` module name -> mlx-audio ``ParakeetTDT`` module path.

    Only the 264 quantised encoder modules are handled here; the dense tensors
    go through the shared full name map.
    """
    if not hf_name.startswith("encoder.layers."):
        raise KeyError(hf_name)
    parts = hf_name.split(".")
    layer = parts[2]
    tail = ".".join(parts[3:])
    attn = {
        "self_attn.q_proj": "self_attn.linear_q",
        "self_attn.k_proj": "self_attn.linear_k",
        "self_attn.v_proj": "self_attn.linear_v",
        "self_attn.o_proj": "self_attn.linear_out",
        "self_attn.relative_k_proj": "self_attn.linear_pos",
    }
    if tail in attn:
        tail = attn[tail]
    elif tail in {
        "feed_forward1.linear1", "feed_forward1.linear2",
        "feed_forward2.linear1", "feed_forward2.linear2",
        "conv.pointwise_conv1", "conv.pointwise_conv2",
    }:
        pass
    else:
        raise KeyError(hf_name)
    return f"encoder.layers.{layer}.{tail}"


# --------------------------------------------------------------------------- planes
def pack_2bit(q: np.ndarray) -> np.ndarray:
    """(O, I) codes in {0..3} -> (O, I//16) uint32, element j of each 16-run in bits 2j..2j+1
    (MLX affine layout, as in v18_runtime / export_v18_packed)."""
    o, i = q.shape
    if i % 16:
        raise ValueError(f"in_features={i} not a multiple of 16")
    q = q.astype(np.uint32).reshape(o, i // 16, 16)
    shifts = (2 * np.arange(16, dtype=np.uint32))[None, None, :]
    return (q << shifts).sum(axis=2, dtype=np.uint32)


def unpack_2bit(packed: np.ndarray, in_features: int) -> np.ndarray:
    o, w = packed.shape
    shifts = (2 * np.arange(16, dtype=np.uint32))[None, None, :]
    q = (packed[:, :, None] >> shifts) & 3
    return q.reshape(o, w * 16)[:, :in_features].astype(np.uint8)


@dataclass
class FiveValueRecord:
    """Raw five-value record as the container stores it (before any expansion)."""
    name: str                # HF module name
    sign: np.ndarray         # (O, I) int8 in {-1, 0, +1}
    is_hi: np.ndarray        # (O, I) bool, True only where sign != 0
    lo: np.ndarray           # (O,) float16
    hi: np.ndarray           # (O,) float16

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.sign.shape)

    def dense(self, dtype=np.float32) -> np.ndarray:
        """Exact expansion: {0, +-lo, +-hi} in the requested dtype."""
        lo = self.lo.astype(np.float32)[:, None]
        hi = self.hi.astype(np.float32)[:, None]
        mag = np.where(self.is_hi, hi, lo)
        return (self.sign.astype(np.float32) * mag).astype(dtype)

    def planes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """-> (base_q uint32 packed, resid_q uint32 packed, lo fp32 (O,), d fp32 (O,))."""
        sign = self.sign.astype(np.int8)
        if np.any(self.is_hi & (sign == 0)):
            raise ValueError(f"{self.name}: residual bit set on a zero weight")
        q_a = (sign + 1).astype(np.uint8)
        q_b = (1 + sign * self.is_hi.astype(np.int8)).astype(np.uint8)
        lo = self.lo.astype(np.float32)
        d = self.hi.astype(np.float32) - lo
        if np.any(d < 0):
            raise ValueError(f"{self.name}: hi < lo in a row")
        return pack_2bit(q_a), pack_2bit(q_b), lo, d


def expand_planes(base_q, resid_q, lo, d, in_features: int) -> np.ndarray:
    """fp32 dequantisation of the two planes -- the G-PLANE check reference."""
    qa = unpack_2bit(np.asarray(base_q), in_features).astype(np.float32)
    qb = unpack_2bit(np.asarray(resid_q), in_features).astype(np.float32)
    lo = np.asarray(lo, dtype=np.float32)[:, None]
    d = np.asarray(d, dtype=np.float32)[:, None]
    return (lo * qa - lo) + (d * qb - d)


# --------------------------------------------------------------------------- module
class PackedFiveValueLinear(nn.Module):
    """Five-value linear = two native 2-bit affine QMMs with per-row metadata.

    Works for ``nn.Linear`` and for mlx-audio's kernel-1 ``nn.Conv1d`` (channels-last,
    so a kernel-1 conv IS a linear over the last axis).
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False,
                 scale_dtype: mx.Dtype = mx.float32):
        super().__init__()
        if in_features % GROUP:
            raise ValueError(f"in_features={in_features} is not divisible by {GROUP}")
        self.in_features = in_features
        self.out_features = out_features
        self.scale_dtype = scale_dtype
        self.base_q = mx.zeros((out_features, in_features // 16), dtype=mx.uint32)
        self.residual_q = mx.zeros((out_features, in_features // 16), dtype=mx.uint32)
        self.row_lo = mx.zeros((out_features,), dtype=mx.float16)
        self.row_hi = mx.zeros((out_features,), dtype=mx.float16)
        if bias:
            self.bias = mx.zeros((out_features,), dtype=scale_dtype)

    @classmethod
    def from_record(cls, rec: FiveValueRecord, *, bias: mx.array | None = None,
                    scale_dtype: mx.Dtype = mx.float32) -> "PackedFiveValueLinear":
        o, i = rec.shape
        m = cls(i, o, bias=bias is not None, scale_dtype=scale_dtype)
        base_q, resid_q, lo, d = rec.planes()
        m.base_q = mx.array(base_q)
        m.residual_q = mx.array(resid_q)
        m.row_lo = mx.array(rec.lo.astype(np.float16))
        m.row_hi = mx.array(rec.hi.astype(np.float16))
        if bias is not None:
            m.bias = bias.astype(scale_dtype)
        m.materialize_runtime_metadata()
        return m

    def materialize_runtime_metadata(self) -> None:
        """Broadcast per-row lo / d to contiguous per-group scales + biases (QMM needs
        physically contiguous metadata; a stride-0 view is not valid input)."""
        groups = self.in_features // GROUP
        shape = (self.out_features, groups)
        lo = self.row_lo.astype(mx.float32)
        d = self.row_hi.astype(mx.float32) - lo            # exact: two fp16 values in fp32
        lo = lo.astype(self.scale_dtype)
        d = d.astype(self.scale_dtype)
        self._base_scales = mx.contiguous(mx.broadcast_to(lo[:, None], shape))
        self._base_biases = mx.contiguous(-self._base_scales)
        self._resid_scales = mx.contiguous(mx.broadcast_to(d[:, None], shape))
        self._resid_biases = mx.contiguous(-self._resid_scales)
        mx.eval(self._base_scales, self._base_biases, self._resid_scales, self._resid_biases)

    def dequantize(self) -> mx.array:
        """fp32 dense weight (O, I) -- for gates only."""
        a = mx.dequantize(self.base_q, self._base_scales.astype(mx.float32),
                          self._base_biases.astype(mx.float32), group_size=GROUP, bits=2)
        b = mx.dequantize(self.residual_q, self._resid_scales.astype(mx.float32),
                          self._resid_biases.astype(mx.float32), group_size=GROUP, bits=2)
        return a + b

    def __call__(self, x: mx.array) -> mx.array:
        in_dtype = x.dtype
        if self.scale_dtype != in_dtype:
            x = x.astype(self.scale_dtype)
        y = mx.quantized_matmul(x, self.base_q, self._base_scales, self._base_biases,
                                transpose=True, group_size=GROUP, bits=2, mode="affine")
        y = y + mx.quantized_matmul(x, self.residual_q, self._resid_scales, self._resid_biases,
                                    transpose=True, group_size=GROUP, bits=2, mode="affine")
        if "bias" in self:
            y = y + self.bias
        return y.astype(in_dtype)


# --------------------------------------------------------------------------- substitution
def _resolve(root: nn.Module, dotted: str):
    parent = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def substitute_five_value(model: nn.Module, records: Iterable[FiveValueRecord], *,
                          scale_dtype: mx.Dtype = mx.float32,
                          expect_modules: int = 264) -> list[dict]:
    """Replace every quantised Parakeet linear / kernel-1 conv with the packed module.

    Asserts shape agreement with the module being replaced, ``in_features % 128 == 0``
    for every record, and that exactly ``expect_modules`` were substituted.
    """
    receipts = []
    n = 0
    for rec in records:
        mlx_name = five_value_hf_to_mlx(rec.name)
        parent, leaf = _resolve(model, mlx_name)
        old = getattr(parent, leaf)
        o, i = rec.shape
        if i % GROUP:
            raise ValueError(f"{rec.name}: in_features {i} % {GROUP} != 0")
        if isinstance(old, nn.Linear):
            assert tuple(old.weight.shape) == (o, i), (rec.name, old.weight.shape, (o, i))
            bias = getattr(old, "bias", None) if "bias" in old else None
            kind = "linear"
        elif isinstance(old, nn.Conv1d):
            assert tuple(old.weight.shape) == (o, 1, i), (rec.name, old.weight.shape, (o, 1, i))
            bias = getattr(old, "bias", None) if "bias" in old else None
            kind = "conv1d_k1"
        else:
            raise TypeError(f"{mlx_name}: unexpected module {type(old).__name__}")
        new = PackedFiveValueLinear.from_record(rec, bias=bias, scale_dtype=scale_dtype)
        setattr(parent, leaf, new)
        n += 1
        receipts.append({"hf": rec.name, "mlx": mlx_name, "kind": kind, "shape": [o, i]})
    if n != expect_modules:
        raise RuntimeError(f"substituted {n} modules, expected {expect_modules}")
    return receipts


def plane_gate(model: nn.Module, records: Iterable[FiveValueRecord]) -> dict:
    """G-PLANE: dequantised planes == exact five-value expansion, per module."""
    worst = 0.0
    rows = []
    for rec in records:
        parent, leaf = _resolve(model, five_value_hf_to_mlx(rec.name))
        mod = getattr(parent, leaf)
        deq = np.asarray(mod.dequantize().astype(mx.float32))
        ref = rec.dense(np.float32)
        delta = float(np.abs(deq - ref).max())
        worst = max(worst, delta)
        distinct = int(np.unique(np.abs(ref)).size)
        rows.append({"module": rec.name, "max_abs_delta": delta, "distinct_abs_values_in_module": distinct})
    return {"modules": len(rows), "max_abs_delta": worst, "per_module": rows}
