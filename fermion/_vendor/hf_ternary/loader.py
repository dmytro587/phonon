"""TRTC v4 parser, donor construction, and Transformers Auto registration."""
from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, BitNetConfig,
                          BitNetForCausalLM, GenerationConfig, PretrainedConfig,
                          PreTrainedModel, Qwen2Config, Qwen2ForCausalLM,
                          Qwen3Config, Qwen3ForCausalLM)

from .modules import PackedInt8Embedding, PackedInt8Linear, TritLinearInference

MAGIC = 0x43545254
HEADER = struct.Struct("<Iii9i2fi")
PROJECTIONS = (
    ("self_attn", "q_proj"), ("self_attn", "k_proj"),
    ("self_attn", "v_proj"), ("self_attn", "o_proj"),
    ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
)


@dataclass(frozen=True)
class Header:
    version: int
    arch: int
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rotary_dims: int
    intermediate: int
    vocab: int
    max_position: int
    rms_norm_eps: float
    rope_theta: float
    eok: int


class Reader:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_COPY)
        self.off = 0

    def close(self) -> None:
        self.mm.close()
        self.file.close()

    def unpack(self, fmt: str):
        values = struct.unpack_from(fmt, self.mm, self.off)
        self.off += struct.calcsize(fmt)
        return values

    def tensor(self, dtype: torch.dtype, count: int) -> torch.Tensor:
        sizes = {torch.uint8: 1, torch.int8: 1, torch.int32: 4, torch.float32: 4}
        out = torch.frombuffer(self.mm, dtype=dtype, count=count, offset=self.off).clone()
        self.off += count * sizes[dtype]
        return out

    def skip(self, count: int) -> None:
        self.off += count

    def trit_record(self) -> dict[str, Any]:
        rows, cols, cols_pad, stride = self.unpack("<4i")
        if rows <= 0 or cols <= 0 or cols_pad < cols or stride * 8 < cols_pad:
            raise ValueError(f"invalid trit geometry at {self.off - 16}")
        planes = [self.tensor(torch.uint8, rows * stride).reshape(rows, stride)
                  for _ in range(3)]
        s_lo = self.tensor(torch.float32, rows)
        s_hi = self.tensor(torch.float32, rows)
        self.skip(rows * 8)  # rs_lo, rs_hi: writer integrity aids, not inference state
        (has_bias,) = self.unpack("<i")
        bias = self.tensor(torch.float32, rows) if has_bias else None
        return {"rows": rows, "cols": cols, "cols_pad": cols_pad, "stride": stride,
                "bp": planes[0], "bn": planes[1], "br": planes[2],
                "s_lo": s_lo, "s_hi": s_hi, "bias": bias}

    def int8_record(self) -> dict[str, Any]:
        rows, cols, cols_pad, stride = self.unpack("<4i")
        if rows <= 0 or cols <= 0 or stride != cols_pad or cols_pad < cols:
            raise ValueError(f"invalid int8 geometry at {self.off - 16}")
        q = self.tensor(torch.int8, rows * stride).reshape(rows, stride)
        scale = self.tensor(torch.float32, rows)
        self.skip(rows * 4)  # wsum is not needed on signed PyTorch paths
        return {"rows": rows, "cols": cols, "q": q, "scale": scale}


def read_header(path: str | os.PathLike[str]) -> Header:
    with open(path, "rb") as f:
        raw = f.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError("truncated TRTC header")
    h = HEADER.unpack(raw)
    if h[0] != MAGIC or h[1] != 4 or h[2] not in (1, 2, 3):
        raise ValueError(f"need TRTC v4 arch 1/2/3, got magic={h[0]:#x} v={h[1]} arch={h[2]}")
    if h[14] not in (1, 2):
        raise ValueError(f"v4 requires int8 embed lane eok 1/2, got {h[14]}")
    return Header(h[1], h[2], *h[3:15])


def _config(h: Header) -> PretrainedConfig:
    common = dict(
        vocab_size=h.vocab, hidden_size=h.hidden, intermediate_size=h.intermediate,
        num_hidden_layers=h.n_layers, num_attention_heads=h.n_heads,
        num_key_value_heads=h.n_kv_heads, max_position_embeddings=h.max_position,
        rms_norm_eps=h.rms_norm_eps, tie_word_embeddings=(h.eok == 1),
        rope_parameters={"rope_type": "default", "rope_theta": h.rope_theta},
        use_cache=True,
    )
    if h.arch == 1:
        return Qwen2Config(**common, layer_types=["full_attention"] * h.n_layers)
    if h.arch == 2:
        return BitNetConfig(**common, hidden_act="relu2", attention_bias=False,
                            bos_token_id=128000 if h.vocab > 128000 else None,
                            eos_token_id=128001 if h.vocab > 128001 else None)
    return Qwen3Config(**common, head_dim=h.head_dim, attention_bias=False,
                       layer_types=["full_attention"] * h.n_layers,
                       bos_token_id=151643 if h.vocab > 151643 else None,
                       eos_token_id=151645 if h.vocab > 151645 else None,
                       pad_token_id=151643 if h.vocab > 151643 else None)


def _new_model(h: Header) -> PreTrainedModel:
    config = _config(h)
    cls = {1: Qwen2ForCausalLM, 2: BitNetForCausalLM, 3: Qwen3ForCausalLM}[h.arch]
    with torch.device("meta"):
        model = cls(config)
    # Rotary frequency buffers are derived state, not container weights.
    model.model.rotary_emb = type(model.model.rotary_emb)(config, device="cpu")
    return model


def _set_weight(module: nn.Module, value: torch.Tensor, dtype: torch.dtype) -> None:
    module._parameters["weight"] = nn.Parameter(value.to(dtype=dtype), requires_grad=False)


def _trit_module(rec: dict[str, Any], row_chunk: int,
                 activation_quant: bool = False) -> TritLinearInference:
    return TritLinearInference(rec["bp"], rec["bn"], rec["br"], rec["s_lo"],
                               rec["s_hi"], rec["cols"], rec["bias"], row_chunk,
                               activation_quant=activation_quant)


def load_trtc_v4(path: str | os.PathLike[str], *, dtype: torch.dtype | None = None,
                 row_chunk: int = 512, int8_row_chunk: int = 4096) -> PreTrainedModel:
    """Load a v4 arch-1/2/3 container as a standard Transformers model.

    Stored s_lo/s_hi are applied directly, matching the nominal expander
    function. Unlike the int8 tensor-core grid, this PyTorch reference has no
    127-byte table cap, so wide-ratio rows are never silently clamped.
    """
    path = Path(path).expanduser().resolve()
    h = read_header(path)
    # BitNet's ReLU^2 intermediate requires bf16's exponent range; fp16 can
    # overflow on the first layer. Qwen expander identity is gated in fp16.
    dtype = dtype or (torch.bfloat16 if h.arch == 2 else torch.float16)
    r = Reader(path)
    try:
        r.off = HEADER.size
        embed_in_rec = r.int8_record() if h.eok == 2 else None
        model = _new_model(h)
        layers = model.model.layers
        for index, layer in enumerate(layers):
            _set_weight(layer.input_layernorm, r.tensor(torch.float32, h.hidden), dtype)
            _set_weight(layer.post_attention_layernorm, r.tensor(torch.float32, h.hidden), dtype)
            if h.arch == 2:
                _set_weight(layer.self_attn.attn_sub_norm,
                            r.tensor(torch.float32, h.hidden), dtype)
                _set_weight(layer.mlp.ffn_sub_norm,
                            r.tensor(torch.float32, h.intermediate), dtype)
            elif h.arch == 3:
                _set_weight(layer.self_attn.q_norm,
                            r.tensor(torch.float32, h.head_dim), dtype)
                _set_weight(layer.self_attn.k_norm,
                            r.tensor(torch.float32, h.head_dim), dtype)
            for parent_name, child_name in PROJECTIONS:
                rec = r.trit_record()
                parent = getattr(layer, parent_name)
                old = getattr(parent, child_name)
                if (old.in_features, old.out_features) != (rec["cols"], rec["rows"]):
                    raise ValueError(f"layer {index} {parent_name}.{child_name}: header/model "
                                     f"shape disagrees with record {rec['rows']}x{rec['cols']}")
                setattr(parent, child_name, _trit_module(
                    rec, row_chunk, activation_quant=(h.arch == 2)))
        _set_weight(model.model.norm, r.tensor(torch.float32, h.hidden), dtype)
        embed_out_rec = r.int8_record()
        if r.off != len(r.mm):
            raise ValueError(f"container walk ended at {r.off}, file size is {len(r.mm)}")
        if embed_out_rec["rows"] != h.vocab or embed_out_rec["cols"] != h.hidden:
            raise ValueError("embed_out record disagrees with header")
        if embed_in_rec is None:
            embed_in_rec = embed_out_rec
        if embed_in_rec["rows"] != h.vocab or embed_in_rec["cols"] != h.hidden:
            raise ValueError("embed_in record disagrees with header")
        model.model.embed_tokens = PackedInt8Embedding(
            embed_in_rec["q"], embed_in_rec["scale"], h.hidden, output_dtype=dtype)
        model.lm_head = PackedInt8Linear(
            embed_out_rec["q"], embed_out_rec["scale"], h.hidden,
            row_chunk=int8_row_chunk, output_dtype=dtype)
        model.eval()
        model.config.trtc_v4_source = str(path)
        model.config.trtc_v4_arch = h.arch
        return model
    finally:
        r.close()


class TrtcV4Config(PretrainedConfig):
    model_type = "trtc_v4"

    def __init__(self, container: str = "model.bin", dtype=None, trtc_dtype: str | None = None,
                 row_chunk: int = 512, int8_row_chunk: int = 4096, **kwargs):
        # The container's preferred compute dtype lives in `trtc_dtype` ("auto" = the loader's per-arch default). It is
        # deliberately NOT `PretrainedConfig.dtype`: transformers >= 5 converts any string there with getattr(torch, s)
        # in __post_init__, so the old `dtype="auto"` default raised AttributeError: module 'torch' has no attribute
        # 'auto' as soon as AutoConfig/AutoTokenizer read a saved config (0.2.0, every bare `fermion serve`). A saved
        # config.json that still carries "dtype": "auto" (or a real dtype name) is accepted and mapped here.
        legacy = kwargs.pop("torch_dtype", None)
        if trtc_dtype is None:
            trtc_dtype = dtype if isinstance(dtype, str) else (legacy if isinstance(legacy, str) else "auto")
        real = None
        for cand in (dtype, legacy):
            if cand is not None and not isinstance(cand, str):
                real = cand
            elif isinstance(cand, str) and cand != "auto":
                real = cand                     # a genuine dtype name: let transformers keep it as the model dtype
        super().__init__(**({"dtype": real} if real is not None else {}), **kwargs)
        self.container = container
        self.trtc_dtype = trtc_dtype
        self.row_chunk = row_chunk
        self.int8_row_chunk = int8_row_chunk


class TrtcV4ForCausalLM(PreTrainedModel):
    config_class = TrtcV4Config

    def __init__(self, config: TrtcV4Config):
        super().__init__(config)
        raise RuntimeError("use from_pretrained() so the config directory can resolve its container")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, config=None, **kwargs):
        root = Path(pretrained_model_name_or_path).expanduser().resolve()
        if config is None:
            config = TrtcV4Config.from_pretrained(root)
        unsupported = set(kwargs) - {"torch_dtype", "dtype", "local_files_only",
                                     "trust_remote_code", "adapter_kwargs"}
        if unsupported:
            raise TypeError(f"unsupported phase-1 load options: {sorted(unsupported)}")
        if kwargs.get("adapter_kwargs"):
            raise TypeError("phase-1 loader does not support adapters")
        cfg_dtype = getattr(config, "trtc_dtype", None) or "auto"
        dtype_arg = kwargs.get("dtype", kwargs.get("torch_dtype", cfg_dtype))
        if dtype_arg is None:
            dtype_arg = cfg_dtype
        dtype = None if dtype_arg == "auto" else (
            dtype_arg if isinstance(dtype_arg, torch.dtype) else getattr(torch, str(dtype_arg).replace("torch.", "")))
        container = Path(config.container).expanduser()
        if not container.is_absolute():
            container = root / container
        model = load_trtc_v4(container, dtype=dtype, row_chunk=config.row_chunk,
                             int8_row_chunk=config.int8_row_chunk)
        # This override replaces PreTrainedModel.from_pretrained wholesale, so
        # nothing else picks the repo's generation_config.json up. Without
        # this, every generation_config.json we ship is inert on the torch
        # path and a bare `model.generate()` silently runs HF library
        # defaults (temperature 1.0, top-k 50, no repetition penalty) instead
        # of the documented config. Measured 2026-07-27.
        try:
            model.generation_config = GenerationConfig.from_pretrained(root)
        except (OSError, ValueError):
            pass                       # no generation_config.json in the repo
        return model


def register() -> None:
    """Register the local config/model pair with Transformers Auto classes."""
    try:
        AutoConfig.register(TrtcV4Config.model_type, TrtcV4Config)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(TrtcV4Config, TrtcV4ForCausalLM)
    except ValueError:
        pass


def write_transformers_config(container: str | os.PathLike[str],
                              output_dir: str | os.PathLike[str], **kwargs) -> Path:
    """Write a tiny local config directory consumed by AutoModelForCausalLM."""
    container = Path(container).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(container, output)
    config = TrtcV4Config(container=rel, **kwargs)
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    return output
