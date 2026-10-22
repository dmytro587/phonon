"""Transformers integration for packed TRTC v4 ternary containers."""
from .loader import (TrtcV4Config, TrtcV4ForCausalLM, load_trtc_v4, read_header,
                     register, write_transformers_config)
from .modules import (PackedInt8Embedding, PackedInt8Linear,
                      TritLinearInference, resident_buffer_bytes)

register()

__all__ = [
    "TritLinearInference", "PackedInt8Embedding", "PackedInt8Linear",
    "TrtcV4Config", "TrtcV4ForCausalLM", "load_trtc_v4", "read_header",
    "register", "write_transformers_config", "resident_buffer_bytes",
]
