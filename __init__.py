"""
ComfyUI-QwenImageWanBridge
Qwen2.5-VL implementation with custom vision support
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Web directory for JavaScript extensions
WEB_DIRECTORY = "./web"

# ============================================================================
# CORE NODES - Qwen2.5-VL with proper vision support
# ============================================================================

try:
    from .nodes.qwen_vl_encoder import QwenVLCLIPLoader, QwenVLTextEncoder
    NODE_CLASS_MAPPINGS["QwenVLCLIPLoader"] = QwenVLCLIPLoader
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLCLIPLoader"] = "Qwen2.5-VL CLIP Loader"

    NODE_CLASS_MAPPINGS["QwenVLTextEncoder"] = QwenVLTextEncoder
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLTextEncoder"] = "Qwen2.5-VL Text Encoder"

    print("[QwenImageWanBridge] Loaded Qwen2.5-VL nodes using ComfyUI's CLIP")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load encoder nodes: {e}")

# Helper nodes for 16-channel latents
try:
    from .nodes.qwen_vl_text_encoder import (
        QwenVLEmptyLatent,
        QwenVLImageToLatent
    )

    NODE_CLASS_MAPPINGS["QwenVLEmptyLatent"] = QwenVLEmptyLatent
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLEmptyLatent"] = "Qwen Empty Latent (16ch)"

    NODE_CLASS_MAPPINGS["QwenVLImageToLatent"] = QwenVLImageToLatent
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLImageToLatent"] = "Qwen Image to Latent (16ch)"

    print("[QwenImageWanBridge] Loaded Qwen helper nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Qwen helper nodes: {e}")

# Resolution helper nodes
try:
    from .nodes.qwen_resolution_helper import (
        QwenOptimalResolution,
        QwenResolutionSelector
    )

    NODE_CLASS_MAPPINGS["QwenOptimalResolution"] = QwenOptimalResolution
    NODE_DISPLAY_NAME_MAPPINGS["QwenOptimalResolution"] = "Qwen Optimal Resolution"

    NODE_CLASS_MAPPINGS["QwenResolutionSelector"] = QwenResolutionSelector
    NODE_DISPLAY_NAME_MAPPINGS["QwenResolutionSelector"] = "Qwen Resolution Selector"

    print("[QwenImageWanBridge] Loaded resolution helper nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load resolution nodes: {e}")

# Template builder nodes for custom prompting
try:
    from .nodes.qwen_template_builder import (
        QwenTemplateBuilder,
        QwenTokenInfo,
        QwenPromptFormatter
    )

    NODE_CLASS_MAPPINGS["QwenTemplateBuilder"] = QwenTemplateBuilder
    NODE_DISPLAY_NAME_MAPPINGS["QwenTemplateBuilder"] = "Qwen Template Builder"

    NODE_CLASS_MAPPINGS["QwenTokenInfo"] = QwenTokenInfo
    NODE_DISPLAY_NAME_MAPPINGS["QwenTokenInfo"] = "Qwen Token Reference"

    NODE_CLASS_MAPPINGS["QwenPromptFormatter"] = QwenPromptFormatter
    NODE_DISPLAY_NAME_MAPPINGS["QwenPromptFormatter"] = "Qwen Prompt Formatter"

    print("[QwenImageWanBridge] Loaded template builder nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load template nodes: {e}")

# Simplified template builder V2
try:
    from .nodes.qwen_template_builder_v2 import (
        QwenTemplateBuilderV2,
        QwenTemplateConnector
    )

    NODE_CLASS_MAPPINGS["QwenTemplateBuilderV2"] = QwenTemplateBuilderV2
    NODE_DISPLAY_NAME_MAPPINGS["QwenTemplateBuilderV2"] = "Qwen Template Builder V2"

    NODE_CLASS_MAPPINGS["QwenTemplateConnector"] = QwenTemplateConnector
    NODE_DISPLAY_NAME_MAPPINGS["QwenTemplateConnector"] = "Qwen Template Connector"

    print("[QwenImageWanBridge] Loaded template builder V2 nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load template V2 nodes: {e}")

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[QwenImageWanBridge] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
