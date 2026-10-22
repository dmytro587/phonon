# Derived from transformers/models/parakeet/configuration_parakeet.py (Hugging Face, Apache-2.0, v5.18).
# Plain classes: no Hub, no serialisation; unknown keys are kept as attributes, as the original does.
from __future__ import annotations


class ParakeetEncoderConfig:
    model_type = "parakeet_encoder"

    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4096, num_hidden_layers: int = 24,
                 num_attention_heads: int = 8, num_key_value_heads: int | None = None, num_mel_bins: int = 128,
                 conv_kernel_size: int = 9, subsampling_conv_channels: int = 256, subsampling_conv_kernel_size: int = 3,
                 subsampling_conv_stride: int = 2, subsampling_factor: int = 8, max_position_embeddings: int = 5000,
                 scale_input: bool = False, attention_bias: bool = False, convolution_bias: bool = False,
                 hidden_act: str = "silu", dropout: float = 0.0, attention_dropout: float = 0.0,
                 activation_dropout: float = 0.0, dropout_positions: float = 0.0, layerdrop: float = 0.0,
                 initializer_range: float = 0.02, **kwargs):
        self.hidden_size = hidden_size; self.intermediate_size = intermediate_size; self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads; self.num_mel_bins = num_mel_bins; self.conv_kernel_size = conv_kernel_size
        self.subsampling_conv_channels = subsampling_conv_channels; self.subsampling_conv_kernel_size = subsampling_conv_kernel_size
        self.subsampling_conv_stride = subsampling_conv_stride; self.subsampling_factor = subsampling_factor
        self.max_position_embeddings = max_position_embeddings; self.scale_input = scale_input; self.attention_bias = attention_bias
        self.convolution_bias = convolution_bias; self.hidden_act = hidden_act; self.dropout = dropout
        self.attention_dropout = attention_dropout; self.activation_dropout = activation_dropout
        self.dropout_positions = dropout_positions; self.layerdrop = layerdrop; self.initializer_range = initializer_range
        self.num_key_value_heads = num_attention_heads               # as the original's __post_init__
        for k, v in kwargs.items():
            setattr(self, k, v)


class ParakeetTDTConfig:
    model_type = "parakeet_tdt"

    def __init__(self, vocab_size: int = 8193, decoder_hidden_size: int = 640, num_decoder_layers: int = 2,
                 hidden_act: str = "relu", max_symbols_per_step: int = 10, blank_token_id: int = 8192,
                 pad_token_id: int = 2, durations=(0, 1, 2, 3, 4), encoder_config=None, initializer_range: float = 0.02,
                 **kwargs):
        self.vocab_size = vocab_size; self.decoder_hidden_size = decoder_hidden_size; self.num_decoder_layers = num_decoder_layers
        self.hidden_act = hidden_act; self.max_symbols_per_step = max_symbols_per_step; self.blank_token_id = blank_token_id
        self.pad_token_id = pad_token_id; self.durations = list(durations)
        if isinstance(encoder_config, dict):
            encoder_config = ParakeetEncoderConfig(**encoder_config)
        self.encoder_config = encoder_config if encoder_config is not None else ParakeetEncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        for k, v in kwargs.items():
            setattr(self, k, v)
