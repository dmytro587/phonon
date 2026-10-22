"""Parakeet-TDT model classes on plain PyTorch (0.2.8).

A torch-only rendering of the Parakeet Fast-Conformer encoder, LSTM prediction network and TDT joint network from
Hugging Face `transformers` (`models/parakeet`, v5.18, Apache-2.0; architecture by NVIDIA NeMo). Same module tree, same
parameter names and shapes, so the Phonon-2 containers load into it unchanged. The CPU engine uses it wherever the
installed `transformers` lacks `ParakeetForTDT` or has switched PyTorch off: Intel Macs, whose last torch build is 2.2.2,
which `transformers` 5.1 and later refuse. Inference only (no training, generation or Hub code).
"""
from .configuration import ParakeetEncoderConfig, ParakeetTDTConfig
from .modeling import (ParakeetEncoder, ParakeetEncoderRelPositionalEncoding, ParakeetForTDT, ParakeetRNNTDecoder,
                       ParakeetTDTJointNetwork)

__all__ = ["ParakeetEncoderConfig", "ParakeetTDTConfig", "ParakeetEncoder", "ParakeetEncoderRelPositionalEncoding",
           "ParakeetForTDT", "ParakeetRNNTDecoder", "ParakeetTDTJointNetwork"]
