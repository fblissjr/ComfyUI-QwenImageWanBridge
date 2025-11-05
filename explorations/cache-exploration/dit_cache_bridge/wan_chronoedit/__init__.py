"""
Wan2.1 ↔ ChronoEdit KV Cache Bridging

Highest priority scenario: Bridge between Wan2.1 and ChronoEdit models.

These models have identical architectures (40 layers, 5120 hidden dim,
40 heads, 128 head_dim), making KV cache bridging nearly trivial.

Key benefits:
- Combine Wan2.1's video generation quality with ChronoEdit's temporal consistency
- Minimal projection overhead (identity or learned gating)
- Easy to train and deploy
"""

from .bridge import WanChronoEditBridge
from .config import WanChronoEditConfig

__all__ = [
    "WanChronoEditBridge",
    "WanChronoEditConfig",
]
