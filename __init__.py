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

# Helper nodes
try:
    from .nodes.qwen_vl_helpers import QwenVLEmptyLatent, QwenVLImageToLatent
    
    NODE_CLASS_MAPPINGS["QwenVLEmptyLatent"] = QwenVLEmptyLatent
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLEmptyLatent"] = "Qwen VL Empty Latent"
    
    NODE_CLASS_MAPPINGS["QwenVLImageToLatent"] = QwenVLImageToLatent
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLImageToLatent"] = "Qwen VL Image to Latent"

    print("[QwenImageWanBridge] Loaded Qwen VL helper nodes (2 nodes)")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load helper nodes: {e}")

# Resolution nodes removed - functionality integrated into encoder

# Simplified template builder V2
try:
    from .nodes.qwen_template_builder import (
        QwenTemplateBuilderV2,
        QwenTemplateConnector
    )

    NODE_CLASS_MAPPINGS["QwenTemplateBuilder"] = QwenTemplateBuilderV2
    NODE_DISPLAY_NAME_MAPPINGS["QwenTemplateBuilder"] = "Qwen Template Builder"

    NODE_CLASS_MAPPINGS["QwenTemplateConnector"] = QwenTemplateConnector
    NODE_DISPLAY_NAME_MAPPINGS["QwenTemplateConnector"] = "Qwen Template Connector"

    print("[QwenImageWanBridge] Loaded template builder nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load template nodes: {e}")

# Qwen-WAN Bridge nodes removed - not related to Qwen-Image-Edit

# V2V nodes removed - not core functionality

# Multi-Reference handler deprecated - use Image Batch node instead

# EliGen Entity Control nodes (DiffSynth-Studio feature)
try:
    from .nodes.qwen_eligen_entity_control import (
        QwenEliGenEntityControl,
        QwenEliGenMaskPainter
    )

    NODE_CLASS_MAPPINGS["QwenEliGenEntityControl"] = QwenEliGenEntityControl
    NODE_DISPLAY_NAME_MAPPINGS["QwenEliGenEntityControl"] = "Qwen EliGen Entity Control"

    NODE_CLASS_MAPPINGS["QwenEliGenMaskPainter"] = QwenEliGenMaskPainter
    NODE_DISPLAY_NAME_MAPPINGS["QwenEliGenMaskPainter"] = "EliGen Mask Painter"

    print("[QwenImageWanBridge] Loaded EliGen Entity Control nodes (2 nodes)")
    print("[QwenImageWanBridge] FEATURE: Precise spatial control with masks and per-region prompts")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load EliGen nodes: {e}")

# Token Analysis nodes
try:
    from .nodes.qwen_token_debugger import QwenTokenDebugger
    from .nodes.qwen_token_analyzer_standalone import QwenTokenAnalyzerStandalone
    from .nodes.qwen_spatial_token_generator import QwenSpatialTokenGenerator

    NODE_CLASS_MAPPINGS["QwenTokenDebugger"] = QwenTokenDebugger
    NODE_DISPLAY_NAME_MAPPINGS["QwenTokenDebugger"] = "Qwen Token Debugger"

    NODE_CLASS_MAPPINGS["QwenTokenAnalyzer"] = QwenTokenAnalyzerStandalone
    NODE_DISPLAY_NAME_MAPPINGS["QwenTokenAnalyzer"] = "Qwen Token Analyzer"

    NODE_CLASS_MAPPINGS["QwenSpatialTokenGenerator"] = QwenSpatialTokenGenerator
    NODE_DISPLAY_NAME_MAPPINGS["QwenSpatialTokenGenerator"] = "Qwen Spatial Token Generator"

    print("[QwenImageWanBridge] Loaded Token Analysis nodes (3 nodes)")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Token Analysis nodes: {e}")

# Debug Controller node
try:
    from .nodes.qwen_debug_controller import QwenDebugController

    NODE_CLASS_MAPPINGS["QwenDebugController"] = QwenDebugController
    NODE_DISPLAY_NAME_MAPPINGS["QwenDebugController"] = "Qwen Debug Controller"

    print("[QwenImageWanBridge] Loaded Debug Controller node")
    print("[QwenImageWanBridge] FEATURE: Holistic debugging with performance profiling and log analysis")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Debug Controller: {e}")

# Inpainting nodes removed - not core to image edit

# Note: Experimental multi-frame nodes have been archived
# Use Multi-Reference Handler with "index" mode for multi-frame support

# Patches removed - functionality integrated into main encoder

# Native nodes removed - incomplete implementation

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[QwenImageWanBridge] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

# Engine nodes removed - incomplete implementation

# ============================================================================
# AUTO DEBUG TRACING - Apply patches automatically
# ============================================================================

try:
    from .nodes import debug_patch
    debug_patch.apply_debug_patches()
    print("[QwenImageWanBridge] ✅ Debug patches applied (silent mode by default)")
    print("[QwenImageWanBridge] Use QwenDebugController node or set QWEN_DEBUG_VERBOSE=true for tracing")
except Exception as e:
    print(f"[QwenImageWanBridge] ❌ Debug patches failed: {e}")
