"""
Qwen3-VL → Qwen-Image-Edit Vision Bridge

Inject Qwen3-VL's superior vision understanding into Qwen-Image-Edit pipeline.

New downstream capabilities enabled:
- OCR-aware editing (32 languages vs limited text understanding)
- Enhanced spatial reasoning for complex scene manipulations
- Fine detail preservation via DeepStack multi-level features
- Multi-image consistency with 256K context
"""

__version__ = "0.1.0"

from .cache_extraction import VisionCacheExtractor
from .vision_bridge import VisionCacheBridge
from .enhanced_pipeline import EnhancedQwenImageEdit
from .evaluation import CapabilityEvaluator

__all__ = [
    "VisionCacheExtractor",
    "VisionCacheBridge",
    "EnhancedQwenImageEdit",
    "CapabilityEvaluator",
]
