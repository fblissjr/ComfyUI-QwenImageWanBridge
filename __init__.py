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
    from .nodes.qwen_vl_encoder import QwenVLCLIPLoader, QwenVLTextEncoder, QwenLowresFixNode
    NODE_CLASS_MAPPINGS["QwenVLCLIPLoader"] = QwenVLCLIPLoader
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLCLIPLoader"] = "Qwen2.5-VL CLIP Loader"

    NODE_CLASS_MAPPINGS["QwenVLTextEncoder"] = QwenVLTextEncoder
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLTextEncoder"] = "Qwen2.5-VL Text Encoder"

    NODE_CLASS_MAPPINGS["QwenLowresFixNode"] = QwenLowresFixNode
    NODE_DISPLAY_NAME_MAPPINGS["QwenLowresFixNode"] = "Qwen Lowres Fix (Two-Stage)"

    print("[QwenImageWanBridge] Loaded Qwen2.5-VL nodes using ComfyUI's CLIP")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load encoder nodes: {e}")

# Advanced encoder for power users
try:
    from .nodes.qwen_vl_encoder_advanced import QwenVLTextEncoderAdvanced
    NODE_CLASS_MAPPINGS["QwenVLTextEncoderAdvanced"] = QwenVLTextEncoderAdvanced
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLTextEncoderAdvanced"] = "Qwen2.5-VL Text Encoder (Advanced)"

    print("[QwenImageWanBridge] Loaded Advanced Encoder with weighted resolution control")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load advanced encoder: {e}")

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

# Image batching node (aspect-ratio preserving with v2.6.1 scaling)
try:
    from .nodes.qwen_image_batch import QwenImageBatch

    NODE_CLASS_MAPPINGS["QwenImageBatch"] = QwenImageBatch
    NODE_DISPLAY_NAME_MAPPINGS["QwenImageBatch"] = "Qwen Image Batch"

    print("[QwenImageWanBridge] Loaded Qwen Image Batch node")
    print("[QwenImageWanBridge] FEATURE: Aspect-ratio preserving batching with v2.6.1 scaling modes")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load image batch node: {e}")

# Resolution nodes removed - functionality integrated into encoder

# Mask-based inpainting nodes
try:
    from .nodes.qwen_mask_processor import QwenMaskProcessor
    from .nodes.qwen_inpaint_sampler import QwenInpaintSampler

    NODE_CLASS_MAPPINGS["QwenMaskProcessor"] = QwenMaskProcessor
    NODE_DISPLAY_NAME_MAPPINGS["QwenMaskProcessor"] = "Qwen Mask Processor"

    NODE_CLASS_MAPPINGS["QwenInpaintSampler"] = QwenInpaintSampler
    NODE_DISPLAY_NAME_MAPPINGS["QwenInpaintSampler"] = "Qwen Inpainting Sampler"

    print("[QwenImageWanBridge] Loaded mask-based inpainting nodes (2 nodes)")
    print("[QwenImageWanBridge] FEATURE: Mask-based spatial editing with diffusers patterns")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load mask-based inpainting nodes: {e}")

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

# Inpainting nodes restored - mask-based approach aligns with DiffSynth patterns

# Note: Experimental multi-frame nodes have been archived
# Use Multi-Reference Handler with "index" mode for multi-frame support

# Patches removed - functionality integrated into main encoder

# Native nodes removed - incomplete implementation

# ============================================================================
# WRAPPER LOADER NODES - DiffSynth-style model loading
# ============================================================================

try:
    from .nodes.qwen_wrapper_loaders import (
        QwenImageDiTLoaderWrapper,
        QwenVLTextEncoderLoaderWrapper,
        QwenImageVAELoaderWrapper,
        QwenModelManagerWrapper
    )
    from .nodes.qwen_wrapper_sampler import (
        QwenImageSamplerNode,
    )
    from .nodes.qwen_wrapper_processor import (
        QwenProcessorWrapper,
        QwenProcessedToEmbedding,
    )
    from .nodes.qwen_wrapper_nodes import (
        QwenImageEncodeWrapper,
        QwenImageModelWithEdit,
        QwenImageSamplerWithEdit,
        QwenDebugLatents,
    )

    NODE_CLASS_MAPPINGS["QwenImageDiTLoaderWrapper"] = QwenImageDiTLoaderWrapper
    NODE_DISPLAY_NAME_MAPPINGS["QwenImageDiTLoaderWrapper"] = "Qwen Image DiT Loader (Wrapper)"

    NODE_CLASS_MAPPINGS["QwenVLTextEncoderLoaderWrapper"] = QwenVLTextEncoderLoaderWrapper
    NODE_DISPLAY_NAME_MAPPINGS["QwenVLTextEncoderLoaderWrapper"] = "Qwen2.5-VL Text Encoder Loader (Wrapper)"

    NODE_CLASS_MAPPINGS["QwenImageVAELoaderWrapper"] = QwenImageVAELoaderWrapper
    NODE_DISPLAY_NAME_MAPPINGS["QwenImageVAELoaderWrapper"] = "Qwen 16-Channel VAE Loader (Wrapper)"

    NODE_CLASS_MAPPINGS["QwenModelManagerWrapper"] = QwenModelManagerWrapper
    NODE_DISPLAY_NAME_MAPPINGS["QwenModelManagerWrapper"] = "Qwen Image Edit Pipeline Loader (Wrapper)"

    NODE_CLASS_MAPPINGS["QwenImageSamplerNode"] = QwenImageSamplerNode
    NODE_DISPLAY_NAME_MAPPINGS["QwenImageSamplerNode"] = "Qwen Image Sampler (FlowMatch)"

    # Processor nodes for text/image processing
    NODE_CLASS_MAPPINGS["QwenProcessorWrapper"] = QwenProcessorWrapper
    NODE_DISPLAY_NAME_MAPPINGS["QwenProcessorWrapper"] = "Qwen Processor (Wrapper)"

    NODE_CLASS_MAPPINGS["QwenProcessedToEmbedding"] = QwenProcessedToEmbedding
    NODE_DISPLAY_NAME_MAPPINGS["QwenProcessedToEmbedding"] = "Qwen Processed to Embedding (Wrapper)"

    # Edit latent handling nodes
    NODE_CLASS_MAPPINGS["QwenImageEncodeWrapper"] = QwenImageEncodeWrapper
    NODE_DISPLAY_NAME_MAPPINGS["QwenImageEncodeWrapper"] = "Qwen Image Encode (Edit Latents)"

    # QwenImageCombineLatents removed - use Image Batch node instead

    NODE_CLASS_MAPPINGS["QwenImageModelWithEdit"] = QwenImageModelWithEdit
    NODE_DISPLAY_NAME_MAPPINGS["QwenImageModelWithEdit"] = "Qwen Model with Edit Latents"

    NODE_CLASS_MAPPINGS["QwenImageSamplerWithEdit"] = QwenImageSamplerWithEdit
    NODE_DISPLAY_NAME_MAPPINGS["QwenImageSamplerWithEdit"] = "Qwen Sampler with Edit"

    NODE_CLASS_MAPPINGS["QwenDebugLatents"] = QwenDebugLatents
    NODE_DISPLAY_NAME_MAPPINGS["QwenDebugLatents"] = "Qwen Debug Latents"

    print("[QwenImageWanBridge] Loaded Wrapper nodes (11 nodes)")
    print("[QwenImageWanBridge] FEATURE: DiffSynth-style model loading with proper edit latent handling")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load wrapper loader nodes: {e}")

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
