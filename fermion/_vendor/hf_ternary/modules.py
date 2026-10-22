"""Packed, inference-only PyTorch reference modules for TRTC v4."""
from __future__ import annotations

import torch
from torch import nn


def _unpack_rows(plane: torch.Tensor, cols: int, start: int, end: int) -> torch.Tensor:
    """Little-bit-order packed bytes -> bool rows, without persistent expansion."""
    shifts = torch.arange(8, dtype=torch.uint8, device=plane.device)
    bits = (plane[start:end].unsqueeze(-1) >> shifts) & 1
    return bits.reshape(end - start, -1)[:, :cols].bool()


class TritLinearInference(nn.Module):
    """Five-level ternary linear whose persistent weights remain bit-packed.

    Forward expands one row chunk to temporary {-1,0,+1} operands and applies
    the two stored magnitudes after their separate matmuls. It never creates or
    caches a persistent fp16/bf16/fp32 weight matrix. This correctness-first
    decomposition is deliberately a replaceable seam for a fused native op.
    """

    def __init__(self, bp: torch.Tensor, bn: torch.Tensor, br: torch.Tensor,
                 s_lo: torch.Tensor, s_hi: torch.Tensor, in_features: int,
                 bias: torch.Tensor | None = None, row_chunk: int = 512,
                 activation_quant: bool = False):
        super().__init__()
        if bp.dtype != torch.uint8 or bn.dtype != torch.uint8 or br.dtype != torch.uint8:
            raise TypeError("bitplanes must be torch.uint8")
        if bp.shape != bn.shape or bp.shape != br.shape or bp.ndim != 2:
            raise ValueError("bitplanes must have one identical [rows,stride] shape")
        if s_lo.shape != s_hi.shape or s_lo.numel() != bp.shape[0]:
            raise ValueError("scale count must equal output rows")
        self.in_features = int(in_features)
        self.out_features = int(bp.shape[0])
        self.row_chunk = int(row_chunk)
        self.activation_quant = bool(activation_quant)
        self.register_buffer("bp", bp.contiguous())
        self.register_buffer("bn", bn.contiguous())
        self.register_buffer("br", br.contiguous())
        self.register_buffer("s_lo", s_lo.float().contiguous())
        self.register_buffer("s_hi", s_hi.float().contiguous())
        self.register_buffer("bias", None if bias is None else bias.float().contiguous())
        self.has_high = bool(torch.any(br).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {x.shape[-1]}")
        flat = x.reshape(-1, self.in_features)
        if self.activation_quant:
            # BitNet arch-2 linears include symmetric per-token int8 activation
            # quantization. Qwen arch-1/3 expander semantics deliberately do not.
            amax = flat.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
            scale = torch.tensor(127.0, device=flat.device) / amax
            flat = ((flat.float() * scale).round().clamp(-127, 127) / scale).to(x.dtype)
        pieces = []
        for start in range(0, self.out_features, self.row_chunk):
            end = min(start + self.row_chunk, self.out_features)
            pos = _unpack_rows(self.bp, self.in_features, start, end)
            neg = _unpack_rows(self.bn, self.in_features, start, end)
            high = _unpack_rows(self.br, self.in_features, start, end)
            sign = pos.to(torch.int8) - neg.to(torch.int8)
            low_sign = sign.masked_fill(high, 0).to(dtype=flat.dtype)
            y = torch.matmul(flat, low_sign.transpose(0, 1))
            y = y * self.s_lo[start:end].to(dtype=flat.dtype)
            if self.has_high:
                high_sign = sign.masked_fill(~high, 0).to(dtype=flat.dtype)
                yh = torch.matmul(flat, high_sign.transpose(0, 1))
                y = y + yh * self.s_hi[start:end].to(dtype=flat.dtype)
            if self.bias is not None:
                y = y + self.bias[start:end].to(dtype=flat.dtype)
            pieces.append(y)
        return torch.cat(pieces, dim=-1).reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, packed=True, row_chunk={self.row_chunk}, "
                f"activation_quant={self.activation_quant}")


class PackedInt8Embedding(nn.Module):
    """Per-row int8 embedding lane from a v4 container."""

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, cols: int,
                 output_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.num_embeddings = int(qweight.shape[0])
        self.embedding_dim = int(cols)
        self.output_dtype = output_dtype
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scale", scale.float().contiguous())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        q = self.qweight[:, :self.embedding_dim][ids]
        s = self.scale[ids].unsqueeze(-1)
        return (q.float() * s.float()).to(dtype=self.output_dtype)


class PackedInt8Linear(nn.Module):
    """Per-row int8 output lane, chunk-expanded only during forward."""

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, cols: int,
                 row_chunk: int = 4096, output_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_features = int(cols)
        self.out_features = int(qweight.shape[0])
        self.row_chunk = int(row_chunk)
        self.output_dtype = output_dtype
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scale", scale.float().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        for start in range(0, self.out_features, self.row_chunk):
            end = min(start + self.row_chunk, self.out_features)
            # Match the expander exactly: fp32 integer*scale, then dtype cast.
            weight = (self.qweight[start:end, :self.in_features].float()
                      * self.scale[start:end, None]).to(dtype=x.dtype)
            pieces.append(torch.matmul(x, weight.transpose(0, 1)))
        return torch.cat(pieces, dim=-1).to(self.output_dtype)


def resident_buffer_bytes(module: nn.Module) -> int:
    """Count unique persistent tensor storages reachable as module buffers."""
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for buf in module.buffers():
        if buf.device.type == "meta":
            continue
        storage = buf.untyped_storage()
        key = (buf.device.type, storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total
