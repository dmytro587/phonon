# Derived from transformers/models/parakeet/modeling_parakeet.py (Hugging Face, Apache-2.0, v5.18).
# Inference-only, plain torch: no generation mixins, Hub kernels, output recorders or gradient checkpointing.
# Module names, parameter names and shapes are those of the original, so a state dict loads unchanged.
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .configuration import ParakeetEncoderConfig, ParakeetTDTConfig

ACT2FN = {"relu": nn.ReLU(), "silu": nn.SiLU(), "swish": nn.SiLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}


@dataclass
class ParakeetEncoderModelOutput:
    last_hidden_state: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


class ParakeetEncoderRelPositionalEncoding(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.config = config
        self.register_buffer("inv_freq", self.compute_default_relative_positional_parameters(config), persistent=False)

    @staticmethod
    def compute_default_relative_positional_parameters(config: ParakeetEncoderConfig) -> torch.Tensor:
        base = 10000.0
        return 1.0 / (base ** (torch.arange(0, config.hidden_size, 2, dtype=torch.float) / config.hidden_size))

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor):
        seq_length = hidden_states.shape[1]
        position_ids = torch.arange(seq_length - 1, -seq_length, -1, device=hidden_states.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(hidden_states.shape[0], -1, 1).to(hidden_states.device)
        position_ids_expanded = position_ids[None, None, :].float()
        with torch.autocast(device_type=hidden_states.device.type if hidden_states.device.type != "meta" else "cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            pos_embed = torch.stack([freqs.sin(), freqs.cos()], dim=-1)
            pos_embed = pos_embed.reshape(*pos_embed.shape[:-2], -1)
        return pos_embed.to(dtype=hidden_states.dtype)


class ParakeetEncoderFeedForward(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)
        self.activation_dropout = config.activation_dropout

    def forward(self, hidden_states):
        hidden_states = self.activation(self.linear1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        return self.linear2(hidden_states)


class ParakeetEncoderConvolutionModule(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        channels = config.hidden_size; kernel_size = config.conv_kernel_size
        self.activation = ACT2FN[getattr(config, "hidden_act", "silu")]
        self.padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=config.convolution_bias)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=self.padding, groups=channels, bias=config.convolution_bias)
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=config.convolution_bias)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)
        if attention_mask is not None:
            all_masked_rows = torch.all(~attention_mask, dim=2) if attention_mask.dtype == torch.bool else torch.all(~(attention_mask == 0.0), dim=2)
            hidden_states = hidden_states.masked_fill(all_masked_rows, 0.0)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states.transpose(1, 2)


class ParakeetEncoderAttention(nn.Module):
    """Multi-head attention with relative positional encoding (Transformer-XL style), eager implementation."""

    def __init__(self, config: ParakeetEncoderConfig, layer_idx: int):
        super().__init__()
        self.config = config; self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.relative_k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.bias_u = nn.Parameter(torch.zeros(config.num_attention_heads, self.head_dim))
        self.bias_v = nn.Parameter(torch.zeros(config.num_attention_heads, self.head_dim))

    def forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None):
        input_shape = hidden_states.shape[:-1]; batch_size, seq_length = input_shape
        hidden_shape = (batch_size, seq_length, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        q_u = query_states + self.bias_u.view(1, self.config.num_attention_heads, 1, self.head_dim)
        q_v = query_states + self.bias_v.view(1, self.config.num_attention_heads, 1, self.head_dim)
        relative_key_states = self.relative_k_proj(position_embeddings).view(batch_size, -1, self.config.num_attention_heads, self.head_dim)
        matrix_bd = q_v @ relative_key_states.permute(0, 2, 3, 1)
        matrix_bd = self._rel_shift(matrix_bd)[..., :seq_length] * self.scaling
        if attention_mask is not None:
            matrix_bd = matrix_bd.masked_fill_(attention_mask.logical_not(), torch.finfo(matrix_bd.dtype).min)
        attn_weights = torch.matmul(q_u, key_states.transpose(2, 3)) * self.scaling + matrix_bd
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    def _rel_shift(self, attention_scores):
        batch_size, num_heads, query_length, position_length = attention_scores.shape
        attention_scores = nn.functional.pad(attention_scores, pad=(1, 0))
        attention_scores = attention_scores.view(batch_size, num_heads, -1, query_length)
        return attention_scores[:, :, 1:].view(batch_size, num_heads, query_length, position_length)


class ParakeetEncoderSubsamplingConv2D(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        self.kernel_size = config.subsampling_conv_kernel_size; self.stride = config.subsampling_conv_stride
        self.channels = config.subsampling_conv_channels; self.padding = (self.kernel_size - 1) // 2
        self.num_layers = int(math.log2(config.subsampling_factor))
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(1, self.channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
        self.layers.append(nn.ReLU())
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=self.channels))
            self.layers.append(nn.Conv2d(self.channels, self.channels, kernel_size=1))
            self.layers.append(nn.ReLU())
        out_length = config.num_mel_bins // (self.stride ** self.num_layers)
        self.linear = nn.Linear(config.subsampling_conv_channels * out_length, config.hidden_size, bias=True)

    def _get_output_length(self, input_lengths: torch.Tensor, conv_layer: nn.Conv2d):
        if hasattr(conv_layer, "stride") and conv_layer.stride != (1, 1):
            padding = conv_layer.padding; kernel_size = conv_layer.kernel_size[0]; stride = conv_layer.stride[0]
            return (input_lengths + padding[0] + padding[1] - kernel_size) // stride + 1
        return input_lengths

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden_states = input_features.unsqueeze(1)
        current_lengths = attention_mask.sum(-1) if attention_mask is not None else None
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if isinstance(layer, nn.Conv2d) and attention_mask is not None:
                current_lengths = self._get_output_length(current_lengths, layer)
                channel_mask = torch.arange(hidden_states.shape[2], device=attention_mask.device) < current_lengths[:, None]
                hidden_states = hidden_states * channel_mask[:, None, :, None]
        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[2], -1)
        return self.linear(hidden_states)


class ParakeetEncoderBlock(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig, layer_idx: int | None = None):
        super().__init__()
        self.feed_forward1 = ParakeetEncoderFeedForward(config)
        self.self_attn = ParakeetEncoderAttention(config, layer_idx)
        self.conv = ParakeetEncoderConvolutionModule(config)
        self.feed_forward2 = ParakeetEncoderFeedForward(config)
        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, position_embeddings: torch.Tensor | None = None):
        residual = hidden_states
        hidden_states = residual + 0.5 * self.feed_forward1(self.norm_feed_forward1(hidden_states))
        attn_output, _ = self.self_attn(self.norm_self_att(hidden_states), attention_mask=attention_mask, position_embeddings=position_embeddings)
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.conv(self.norm_conv(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + 0.5 * self.feed_forward2(self.norm_feed_forward2(hidden_states))
        return self.norm_out(hidden_states)


class _SubsamplingLengths:
    """`_get_subsampling_output_length` / `_get_output_attention_mask` as in the original pretrained base class."""

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor):
        encoder_config = getattr(self.config, "encoder_config", self.config)
        kernel_size = encoder_config.subsampling_conv_kernel_size; stride = encoder_config.subsampling_conv_stride
        num_layers = int(math.log2(encoder_config.subsampling_factor))
        add_pad = (kernel_size - 1) // 2 * 2 - kernel_size
        lengths = input_lengths
        for _ in range(num_layers):
            lengths = torch.floor(torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + 1.0)
        return lengths.to(dtype=torch.int)

    def _get_output_attention_mask(self, attention_mask: torch.Tensor, target_length: int | None = None):
        output_lengths = self._get_subsampling_output_length(attention_mask.sum(-1))
        max_length = target_length if target_length is not None else output_lengths.max()
        return torch.arange(max_length, device=attention_mask.device) < output_lengths[:, None]


class ParakeetEncoder(nn.Module, _SubsamplingLengths):
    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout; self.dropout_positions = config.dropout_positions; self.layerdrop = config.layerdrop
        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0
        self.subsampling = ParakeetEncoderSubsamplingConv2D(config)
        self.encode_positions = ParakeetEncoderRelPositionalEncoding(config)
        self.layers = nn.ModuleList([ParakeetEncoderBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None, output_attention_mask: bool = True, **kwargs):
        hidden_states = self.subsampling(input_features, attention_mask) * self.input_scale
        position_embeddings = self.encode_positions(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        position_embeddings = nn.functional.dropout(position_embeddings, p=self.dropout_positions, training=self.training)
        output_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(attention_mask, target_length=hidden_states.shape[1])
            attention_mask = output_mask.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
            attention_mask = (attention_mask & attention_mask.transpose(1, 2)).unsqueeze(1)
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
        return ParakeetEncoderModelOutput(last_hidden_state=hidden_states,
                                          attention_mask=output_mask.int() if attention_mask is not None and output_attention_mask else None)


class ParakeetRNNTDecoder(nn.Module):
    """LSTM prediction network."""

    def __init__(self, config: ParakeetTDTConfig):
        super().__init__()
        self.blank_token_id = config.blank_token_id
        self.embedding = nn.Embedding(config.vocab_size, config.decoder_hidden_size)
        self.lstm = nn.LSTM(input_size=config.decoder_hidden_size, hidden_size=config.decoder_hidden_size, num_layers=config.num_decoder_layers, batch_first=True)
        self.decoder_projector = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)

    def forward(self, input_ids: torch.LongTensor, state=None):
        lstm_output, state = self.lstm(self.embedding(input_ids), state)
        return self.decoder_projector(lstm_output), state


class ParakeetTDTJointNetwork(nn.Module):
    """Joint network: token logits + duration logits (`vocab_size + len(durations)` outputs)."""

    def __init__(self, config: ParakeetTDTConfig):
        super().__init__()
        self.activation = ACT2FN[config.hidden_act]
        self.head = nn.Linear(config.decoder_hidden_size, config.vocab_size + len(config.durations))
        self.vocab_size = config.vocab_size

    def forward(self, decoder_hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        return self.head(self.activation(encoder_hidden_states + decoder_hidden_states))


class ParakeetForTDT(nn.Module, _SubsamplingLengths):
    """Encoder + projector + prediction network + TDT joint. The greedy decode loop lives in the engine that owns the model."""

    def __init__(self, config: ParakeetTDTConfig):
        super().__init__()
        self.config = config
        self.encoder = ParakeetEncoder(config.encoder_config)
        self.encoder_projector = nn.Linear(config.encoder_config.hidden_size, config.decoder_hidden_size)
        self.decoder = ParakeetRNNTDecoder(config)
        self.joint = ParakeetTDTJointNetwork(config)
        self.max_symbols_per_step = config.max_symbols_per_step

    def get_audio_features(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None):
        out = self.encoder(input_features=input_features, attention_mask=attention_mask)
        return out.last_hidden_state, self.encoder_projector(out.last_hidden_state), out.attention_mask
