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

# Qwen to Wan Bridge Nodes
try:
    from .nodes.qwen_wan_keyframe_editor import (
        QwenWANKeyframeEditor,
        QwenWANKeyframeExtractor
    )

    NODE_CLASS_MAPPINGS["QwenWANKeyframeEditor"] = QwenWANKeyframeEditor
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANKeyframeEditor"] = "Qwen-WAN Keyframe Editor"
    
    NODE_CLASS_MAPPINGS["QwenWANKeyframeExtractor"] = QwenWANKeyframeExtractor
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANKeyframeExtractor"] = "Extract Keyframes for Editing"
    
    print("[QwenImageWanBridge] Loaded Qwen-WAN Bridge nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Qwen-WAN Bridge nodes: {e}")

# Minimal Keyframe V2V nodes (Technical Parameters)
try:
    from .nodes.minimal_keyframe_v2v import (
        MinimalKeyframeV2V,
        DenoiseCurveVisualizer,
        LatentStatisticsMonitor
    )
    
    NODE_CLASS_MAPPINGS["MinimalKeyframeV2V"] = MinimalKeyframeV2V
    NODE_DISPLAY_NAME_MAPPINGS["MinimalKeyframeV2V"] = "Minimal Keyframe V2V (Technical)"
    
    NODE_CLASS_MAPPINGS["DenoiseCurveVisualizer"] = DenoiseCurveVisualizer
    NODE_DISPLAY_NAME_MAPPINGS["DenoiseCurveVisualizer"] = "Visualize Denoise Schedule"
    
    NODE_CLASS_MAPPINGS["LatentStatisticsMonitor"] = LatentStatisticsMonitor
    NODE_DISPLAY_NAME_MAPPINGS["LatentStatisticsMonitor"] = "Monitor Latent Statistics"
    
    print("[QwenImageWanBridge] Loaded Minimal Keyframe V2V nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Minimal Keyframe V2V nodes: {e}")

# Multi-Reference nodes
try:
    from .nodes.qwen_multi_reference import QwenMultiReferenceHandler
    
    NODE_CLASS_MAPPINGS["QwenMultiReferenceHandler"] = QwenMultiReferenceHandler
    NODE_DISPLAY_NAME_MAPPINGS["QwenMultiReferenceHandler"] = "Multi-Reference Handler"
    
    print("[QwenImageWanBridge] Loaded Multi-Reference handler")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Multi-Reference handler: {e}")

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[QwenImageWanBridge] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
