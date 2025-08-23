"""
ComfyUI-QwenImageWanBridge
Qwen2.5-VL implementation with proper vision support
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ============================================================================
# CORE NODES - Qwen2.5-VL with proper vision support
# ============================================================================

# Main Qwen2.5-VL Loader and Text Encoder
try:
    from .nodes.qwen_vl_loader import QwenVLLoader
    NODE_CLASS_MAPPINGS["QwenVLLoader"] = QwenVLLoader
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLLoader"] = "Qwen2.5-VL Model Loader"
    
    print("[QwenImageWanBridge] ✓ Loaded Qwen2.5-VL Loader with transformers support")
except Exception as e:
    print(f"[QwenImageWanBridge] ✗ Failed to load Qwen2.5-VL Loader: {e}")

# Proper CLIP-based implementation that actually works
try:
    from .nodes.qwen_vl_encoder import QwenVLCLIPLoader, QwenVLTextEncoderProper
    NODE_CLASS_MAPPINGS["QwenVLCLIPLoader"] = QwenVLCLIPLoader
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLCLIPLoader"] = "Qwen2.5-VL CLIP Loader"
    
    NODE_CLASS_MAPPINGS["QwenVLTextEncoderProper"] = QwenVLTextEncoderProper
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLTextEncoderProper"] = "Qwen2.5-VL Text Encoder"
    
    print("[QwenImageWanBridge] ✓ Loaded proper Qwen2.5-VL nodes using ComfyUI's CLIP")
except Exception as e:
    print(f"[QwenImageWanBridge] ✗ Failed to load proper encoder nodes: {e}")

# Legacy text encoder (has issues with safetensors - returns random noise)
# Keeping for compatibility but use QwenVLTextEncoderProper instead
try:
    from .nodes.qwen_vl_text_encoder import (
        QwenVLTextEncoder,
        QwenVLEmptyLatent,
        QwenVLImageToLatent
    )
    # Commenting out the broken encoder - use QwenVLTextEncoderProper instead
    # NODE_CLASS_MAPPINGS["QwenVLTextEncoder"] = QwenVLTextEncoder
    # NODE_DISPLAY_NAME_MAPPINGS["QwenVLTextEncoder"] = "Qwen2.5-VL Text Encoder (Legacy)"
    
    NODE_CLASS_MAPPINGS["QwenVLEmptyLatent"] = QwenVLEmptyLatent
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLEmptyLatent"] = "Qwen Empty Latent (16ch)"
    
    NODE_CLASS_MAPPINGS["QwenVLImageToLatent"] = QwenVLImageToLatent
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLImageToLatent"] = "Qwen Image to Latent (16ch)"
    
    print("[QwenImageWanBridge] ✓ Loaded Qwen helper nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] ✗ Failed to load Qwen helper nodes: {e}")

# ============================================================================
# OPTIONAL NODES - Comment out if they cause issues
# ============================================================================

# Simplified Qwen Loaders (for backward compatibility)
try:
    from .nodes.qwen_text_encoder_loader import (
        QwenTextEncoderLoader,
        QwenDiffusionModelLoader,
        QwenVAELoader,
        QwenCheckpointLoaderSimple
    )
    NODE_CLASS_MAPPINGS["QwenTextEncoderLoader"] = QwenTextEncoderLoader
    NODE_DISPLAY_NAME_MAPPINGS["QwenTextEncoderLoader"] = "Load Qwen Text Encoder (Legacy)"
    
    NODE_CLASS_MAPPINGS["QwenDiffusionModelLoader"] = QwenDiffusionModelLoader
    NODE_DISPLAY_NAME_MAPPINGS["QwenDiffusionModelLoader"] = "Load Qwen Diffusion Model"
    
    NODE_CLASS_MAPPINGS["QwenVAELoader"] = QwenVAELoader
    NODE_DISPLAY_NAME_MAPPINGS["QwenVAELoader"] = "Load Qwen VAE"
    
    NODE_CLASS_MAPPINGS["QwenCheckpointLoaderSimple"] = QwenCheckpointLoaderSimple
    NODE_DISPLAY_NAME_MAPPINGS["QwenCheckpointLoaderSimple"] = "Load Qwen Checkpoint"
    
    print("[QwenImageWanBridge] ✓ Loaded legacy loader nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] ⚠ Legacy loaders not loaded: {e}")

# ============================================================================
# DEBUG NODES (Optional)
# ============================================================================

try:
    from .nodes.qwen_wan_debug import QwenWANLatentDebug, QwenWANConditioningDebug, QwenWANCompareLatents
    NODE_CLASS_MAPPINGS["QwenWANLatentDebug"] = QwenWANLatentDebug
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANLatentDebug"] = "Latent Debug"
    
    NODE_CLASS_MAPPINGS["QwenWANConditioningDebug"] = QwenWANConditioningDebug
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANConditioningDebug"] = "Conditioning Debug"
    
    NODE_CLASS_MAPPINGS["QwenWANCompareLatents"] = QwenWANCompareLatents
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANCompareLatents"] = "Compare Latents"
    
    print("[QwenImageWanBridge] ✓ Loaded debug nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] ⚠ Debug nodes not loaded: {e}")

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[QwenImageWanBridge] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
print("[QwenImageWanBridge] ============================================")
print("[QwenImageWanBridge] IMPORTANT: Install transformers library:")
print("[QwenImageWanBridge]   pip install transformers")
print("[QwenImageWanBridge] ============================================")
print("[QwenImageWanBridge] Key improvements:")
print("[QwenImageWanBridge] • Vision tokens now actually work")
print("[QwenImageWanBridge] • Proper multimodal model loading")
print("[QwenImageWanBridge] • Exact system prompts from DiffSynth-Studio")
print("[QwenImageWanBridge] • Direct transformers integration")
print("[QwenImageWanBridge] ============================================")