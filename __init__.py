"""
ComfyUI-QwenImageWanBridge
Qwen2.5-VL implementation with proper vision support
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ============================================================================
# CORE NODES - Qwen2.5-VL with proper vision support
# ============================================================================

# Qwen2.5-VL CLIP-based implementation that actually works
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

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[QwenImageWanBridge] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")