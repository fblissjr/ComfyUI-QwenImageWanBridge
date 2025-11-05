"""
DiT Cache Bridging - KV Cache Transfer for Diffusion Transformers

This package implements KV cache bridging techniques for DiT models,
enabling knowledge transfer between different transformer-based models.

Supported scenarios:
1. Wan2.1 ↔ ChronoEdit: High-priority bidirectional bridging
2. Replace UMT5 with Causal LM: Enable LLM→DiT text conditioning
3. Qwen2.5-VL → Qwen3-VL: Version migration with cache transfer
"""

__version__ = "0.1.0"
__author__ = "DiT Cache Bridging Research"

from . import wan_chronoedit
from . import qwen_versions
from . import c2c_adapter
from . import utils

__all__ = [
    "wan_chronoedit",
    "qwen_versions",
    "c2c_adapter",
    "utils",
]
