"""Torch graph assembly for the Phonon packed artifact.

Mounts the official Qwen3-ASR Transformers graph, replaces the 196
text-decoder projections with the packed load-time representation from
`artifact.py`, and loads the remaining audio tower, embedding, norms and
tied vocabulary head from the published safetensors shards.

The model backend (configuration / modeling / processing) is the official
Qwen3-ASR Transformers backend, Apache-2.0, vendored unmodified under
`fermion/_vendor/qwen3_asr_backend/` (see the NOTICE file). An installed
`qwen_asr` package is used as a fallback when the vendored copy is absent,
resolved through a private module alias so that only the three backend
modules load — the upstream package initializer imports aligners, extra
tokenizers and serving helpers this model does not need.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from fermion._speech._engine_cpu.artifact import Fold4Matrix

DECODER_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _transformers_compat() -> None:
    """Let the backend's `@check_model_inputs()` spelling work everywhere.

    The vendored backend calls ``check_model_inputs`` as a decorator FACTORY
    (its transformers-4.57/5.0 signature: ``(func=None, *, ...)``). Newer
    transformers releases make it a plain single-argument decorator, which
    raises at class-creation time on the factory spelling. Wrapping the
    module attribute before the backend imports it accepts both spellings,
    delegating to the installed implementation either way — the vendored
    files stay byte-identical to their upstream.
    """
    import inspect

    from transformers.utils import generic
    fn = generic.check_model_inputs
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return
    if not params or params[0].default is not inspect.Parameter.empty:
        return                       # already accepts the factory spelling

    original = fn

    def check_model_inputs(func=None, **_kwargs):
        if func is None:
            return original
        return original(func)

    generic.check_model_inputs = check_model_inputs
    _causal_mask_compat()


def _causal_mask_compat() -> None:
    """Accept the backend's ``create_causal_mask`` keyword spelling.

    Newer transformers renamed the mask builder's ``input_embeds`` keyword
    to ``inputs_embeds`` and dropped ``cache_position`` (the equivalent
    information now comes from the cache object itself). The backend calls
    with the older spelling; this adapter renames and trims the keywords
    and delegates to the installed implementation.
    """
    import inspect

    from transformers import masking_utils
    fn = masking_utils.create_causal_mask
    try:
        accepted = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return
    if "input_embeds" in accepted:
        return                       # already accepts the older spelling

    original_mask_fn = fn

    def create_causal_mask(*args, **kwargs):
        if "input_embeds" in kwargs and "inputs_embeds" in accepted:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return original_mask_fn(*args, **kwargs)

    masking_utils.create_causal_mask = create_causal_mask


def _classic_default_rope(config=None, device=None, seq_len=None, **_kwargs):
    """The original RoPE inverse-frequency formula, spelled out verbatim:
    ``inv_freq = 1 / base**(2i/dim)``, scaling factor 1.0 — deterministic,
    and byte-identical to what the graph was verified with."""
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = (getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (
        torch.arange(0, dim, 2, dtype=torch.int64)
        .to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def _rope_default_compat() -> None:
    """Keep the ``"default"`` rotary-embedding initializer available.

    The backend asks ``ROPE_INIT_FUNCTIONS`` for ``"default"``. Newer
    transformers releases drop that key (the replacement entries compute
    DIFFERENT frequencies), so the classic formula is registered when
    absent.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _classic_default_rope


@contextlib.contextmanager
def _import_noise_to_stderr():
    """Route import-time advisory prints off stdout.

    Newer transformers' docstring checker PRINTS advisory lines to STDOUT
    while model classes are created (at import). stdout belongs to the
    transcript — `fermion transcribe clip.wav > out.txt` must capture
    exactly the words — so anything printed during the backend import is
    re-emitted on stderr instead of being swallowed — except the
    docstring checker's own `[ERROR] … not documented` advisories about
    the vendored backend, which describe no runtime condition and are
    dropped.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield
    kept = [line for line in buffer.getvalue().splitlines(keepends=True)
            if not (line.lstrip().startswith("[ERROR]")
                    and "documented" in line)]
    if kept:
        sys.stderr.write("".join(kept))


def _qwen_backend_modules():
    """(configuration, modeling, processing) — vendored copy first."""
    _transformers_compat()
    _rope_default_compat()
    base = "fermion._vendor.qwen3_asr_backend"
    if importlib.util.find_spec(base) is not None:
        with _import_noise_to_stderr():
            configuration = importlib.import_module(base + ".configuration_qwen3_asr")
            modeling = importlib.import_module(base + ".modeling_qwen3_asr")
            processing = importlib.import_module(base + ".processing_qwen3_asr")
        _patch_backend(configuration, modeling)
        return configuration, modeling, processing
    # Fallback: an installed qwen_asr wheel, backend modules only.
    package = "_phonon_qwen_transformers_backend"
    if package not in sys.modules:
        spec = importlib.util.find_spec("qwen_asr")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(
                "neither the vendored Qwen3-ASR backend nor an installed "
                "qwen_asr package is available")
        root = Path(next(iter(spec.submodule_search_locations)))
        backend = root / "core" / "transformers_backend"
        if not (backend / "modeling_qwen3_asr.py").is_file():
            raise RuntimeError(
                f"official Qwen Transformers backend missing: {backend}")
        module = types.ModuleType(package)
        module.__path__ = [str(backend)]
        module.__package__ = package
        sys.modules[package] = module
    with _import_noise_to_stderr():
        configuration = importlib.import_module(package + ".configuration_qwen3_asr")
        modeling = importlib.import_module(package + ".modeling_qwen3_asr")
        processing = importlib.import_module(package + ".processing_qwen3_asr")
    _patch_backend(configuration, modeling)
    return configuration, modeling, processing


def _patch_backend(configuration, modeling) -> None:
    """Two small seams that keep the backend loadable on newer transformers.

    1. Construction-time config validation: newer transformers validates a
       config DURING ``super().__init__()`` and that validation calls
       ``get_text_config()`` — which the backend's composite config
       implements by reading a sub-config attribute its own ``__init__``
       has not attached yet at that point. Returning the config itself
       inside that window is exactly what the validator handles (no
       ``vocab_size`` -> nothing to validate); after construction the
       original implementation answers every call, unchanged.

    2. Weight init on rotary modules: newer transformers' generic
       ``_init_weights`` re-derives a rotary module's ``inv_freq`` through
       ``module.compute_default_rope_parameters`` when the rope type is
       "default". The backend's rotary class predates that hook, so the
       classic formula is attached — the same values its ``__init__``
       computed, re-copied.
    """
    cls = configuration.Qwen3ASRConfig
    if not getattr(cls, "_fermion_gtc_patched", False):
        original = cls.get_text_config

        def get_text_config(self, *args, **kwargs):
            try:
                return original(self, *args, **kwargs)
            except AttributeError:
                return self

        cls.get_text_config = get_text_config
        cls._fermion_gtc_patched = True

    rope_cls = getattr(modeling, "Qwen3ASRThinkerTextRotaryEmbedding", None)
    if rope_cls is not None and not hasattr(rope_cls,
                                            "compute_default_rope_parameters"):
        rope_cls.compute_default_rope_parameters = staticmethod(
            _classic_default_rope)

    # Older transformers gave every config an implicit ``pad_token_id``
    # (None when the file carries none); newer releases raise
    # AttributeError instead, and the backend reads the attribute
    # unconditionally (treating None as "no pad token"). A class-level
    # None restores exactly the older answer for configs whose file has no
    # value, without touching configs that do.
    for name in ("Qwen3ASRConfig", "Qwen3ASRThinkerConfig",
                 "Qwen3ASRTextConfig", "Qwen3ASRAudioEncoderConfig"):
        config_cls = getattr(configuration, name, None)
        if config_cls is not None and not hasattr(config_cls, "pad_token_id"):
            config_cls.pad_token_id = None


def _unpack_fold4(weights: Fold4Matrix) -> torch.Tensor:
    """Reconstruct the exact load-time packed matrix as CPU BF16."""
    packed = torch.from_numpy(weights.codes)
    codes = torch.empty((weights.rows, weights.cols), dtype=torch.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    scale = torch.from_numpy(weights.scales).to(torch.float32)[:, None]
    center = torch.from_numpy(weights.centers).to(torch.float32)[:, None]
    return ((codes.to(torch.float32) - center) * scale).to(torch.bfloat16)


def _decoder_module_names(model_dir: Path) -> list[str]:
    manifest = json.loads((Path(model_dir) / "packed_manifest.json").read_text())
    modules = [str(row["name"]) for row in manifest.get("modules", [])]
    if len(modules) != 196:
        raise RuntimeError(
            f"expected 196 packed decoder modules, found {len(modules)}")
    bad = [name for name in modules if not name.endswith(DECODER_SUFFIXES)]
    if bad:
        raise RuntimeError(f"unexpected packed module names: {bad[:3]}")
    return modules


def _set_child(root: nn.Module, dotted_name: str, child: nn.Module) -> None:
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], child)


def _artifact_state(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all non-packed tensors and map them into the official wrapper.

    The whole state at once (the historical shape); ``model.py`` streams
    ``iter_artifact_state`` instead."""
    return dict(iter_artifact_state(model_dir))


def iter_artifact_state(model_dir: Path):
    """``(name, tensor)`` pairs of ``_artifact_state``, one at a time."""
    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        if name.endswith((".quint5_q", ".base_alpha", ".residual_scale")):
            continue
        by_shard.setdefault(shard, []).append(name)

    for shard, names in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                # The artifact stores audio-tower convolutions in the source
                # layout [O, H, W, C]; Torch expects [O, C, H, W]. Invert that
                # one transform here.
                if "conv2d" in name and name.endswith(".weight") and tensor.ndim == 4:
                    tensor = tensor.permute(0, 3, 1, 2).contiguous()
                yield "thinker." + name, tensor
