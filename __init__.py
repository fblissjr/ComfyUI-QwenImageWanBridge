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

# Qwen-to-Wan Video Bridge nodes
try:
    from .nodes.qwen_wan_bridge import (
        QwenToWanFirstFrameLatent,
        QwenToWanLatentSaver,
        QwenToWanImageSaver
    )

    NODE_CLASS_MAPPINGS["QwenToWanFirstFrameLatent"] = QwenToWanFirstFrameLatent
    NODE_DISPLAY_NAME_MAPPINGS["QwenToWanFirstFrameLatent"] = "Qwen → Wan First Frame Latent"

    NODE_CLASS_MAPPINGS["QwenToWanLatentSaver"] = QwenToWanLatentSaver
    NODE_DISPLAY_NAME_MAPPINGS["QwenToWanLatentSaver"] = "Save First Frame Latent (Wan)"

    NODE_CLASS_MAPPINGS["QwenToWanImageSaver"] = QwenToWanImageSaver
    NODE_DISPLAY_NAME_MAPPINGS["QwenToWanImageSaver"] = "Save First Frame Image"

    print("[QwenImageWanBridge] Loaded Qwen-to-Wan Video Bridge nodes (3 nodes)")
    print("[QwenImageWanBridge] FEATURE: Image-to-video bridge with first frame conditioning")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Wan bridge nodes: {e}")

# Qwen-to-ChronoEdit Bridge node
try:
    from .nodes.qwen_chronoedit_bridge import QwenToChronoEditBridge

    NODE_CLASS_MAPPINGS["QwenToChronoEditBridge"] = QwenToChronoEditBridge
    NODE_DISPLAY_NAME_MAPPINGS["QwenToChronoEditBridge"] = "Qwen → ChronoEdit Bridge"

    print("[QwenImageWanBridge] Loaded Qwen-to-ChronoEdit Bridge node")
    print("[QwenImageWanBridge] FEATURE: Qwen image editing → ChronoEdit video animation")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load ChronoEdit bridge node: {e}")

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

# Experimental Smart Crop node
try:
    from .nodes.experimental_smart_crop import QwenSmartCrop

    NODE_CLASS_MAPPINGS["QwenSmartCrop"] = QwenSmartCrop
    NODE_DISPLAY_NAME_MAPPINGS["QwenSmartCrop"] = "Qwen Smart Crop (Experimental)"

    print("[QwenImageWanBridge] Loaded Experimental Smart Crop node")
    print("[QwenImageWanBridge] EXPERIMENTAL: Intelligent face cropping with VLM detection support")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Smart Crop node: {e}")

# C2C Vision Bridge nodes archived - see nodes/archive/c2c/

# Inpainting nodes restored - mask-based approach aligns with DiffSynth patterns

# Note: Experimental multi-frame nodes have been archived
# Use Multi-Reference Handler with "index" mode for multi-frame support

# Patches removed - functionality integrated into main encoder

# Native nodes removed - incomplete implementation

# Wrapper nodes archived - see nodes/archive/wrapper/ (not production ready, had VRAM issues)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[QwenImageWanBridge] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

# Engine nodes removed - incomplete implementation

# ============================================================================
# HUNYUANVIDEO 1.5 NODES - Dual text encoder (Qwen2.5-VL + byT5)
# ============================================================================

# Loader nodes
try:
    from .nodes.hunyuan_video_nodes import (
        HunyuanVideoCLIPLoader,
        HunyuanVideoVisionLoader,
        HunyuanVideoEmptyLatent,
    )

    NODE_CLASS_MAPPINGS["HunyuanVideoCLIPLoader"] = HunyuanVideoCLIPLoader
    NODE_DISPLAY_NAME_MAPPINGS["HunyuanVideoCLIPLoader"] = "HunyuanVideo CLIP Loader"

    NODE_CLASS_MAPPINGS["HunyuanVideoVisionLoader"] = HunyuanVideoVisionLoader
    NODE_DISPLAY_NAME_MAPPINGS["HunyuanVideoVisionLoader"] = "HunyuanVideo Vision Loader (SigLIP)"

    NODE_CLASS_MAPPINGS["HunyuanVideoEmptyLatent"] = HunyuanVideoEmptyLatent
    NODE_DISPLAY_NAME_MAPPINGS["HunyuanVideoEmptyLatent"] = "HunyuanVideo Empty Latent"

    print("[QwenImageWanBridge] Loaded HunyuanVideo 1.5 loader nodes (3 nodes)")
    print("[QwenImageWanBridge] FEATURE: Dual CLIP loader (Qwen2.5-VL + byT5)")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load HunyuanVideo loader nodes: {e}")

# Text encoder with template system
try:
    from .nodes.hunyuan_video_encoder import HunyuanVideoTextEncoder, get_templates as get_hunyuan_video_templates

    NODE_CLASS_MAPPINGS["HunyuanVideoTextEncoder"] = HunyuanVideoTextEncoder
    NODE_DISPLAY_NAME_MAPPINGS["HunyuanVideoTextEncoder"] = "HunyuanVideo Text Encoder"

    # Register API endpoint for templates (single source of truth)
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get("/api/hunyuan_video_templates")
        async def get_hunyuan_video_templates_api(request):
            """API endpoint for HunyuanVideo templates - JS fetches from here."""
            templates = get_hunyuan_video_templates()
            return web.json_response(templates)

        print("[QwenImageWanBridge] Loaded HunyuanVideo text encoder")
        print("[QwenImageWanBridge] API: /api/hunyuan_video_templates endpoint registered")
    except Exception as api_err:
        print(f"[QwenImageWanBridge] Loaded HunyuanVideo encoder (API endpoint failed: {api_err})")

except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load HunyuanVideo encoder: {e}")

# Prompt expander (lightweight alternative to Qwen3-235B rewriter)
try:
    from .nodes.hunyuan_video_prompt_expander import HunyuanVideoPromptExpander

    NODE_CLASS_MAPPINGS["HunyuanVideoPromptExpander"] = HunyuanVideoPromptExpander
    NODE_DISPLAY_NAME_MAPPINGS["HunyuanVideoPromptExpander"] = "HunyuanVideo Prompt Expander"

    print("[QwenImageWanBridge] Loaded HunyuanVideo prompt expander")
    print("[QwenImageWanBridge] FEATURE: Lightweight prompt expansion using loaded Qwen2.5-VL")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load HunyuanVideo prompt expander: {e}")

# ============================================================================
# Z-IMAGE NODES - Proper thinking token support
# ============================================================================

try:
    from .nodes.z_image_encoder import ZImageTextEncoder, ZImageTurnBuilder, PromptKeyFilter, get_templates

    NODE_CLASS_MAPPINGS["ZImageTextEncoder"] = ZImageTextEncoder
    NODE_DISPLAY_NAME_MAPPINGS["ZImageTextEncoder"] = "Z-Image Text Encoder"

    NODE_CLASS_MAPPINGS["ZImageTurnBuilder"] = ZImageTurnBuilder
    NODE_DISPLAY_NAME_MAPPINGS["ZImageTurnBuilder"] = "Z-Image Turn Builder"

    NODE_CLASS_MAPPINGS["PromptKeyFilter"] = PromptKeyFilter
    NODE_DISPLAY_NAME_MAPPINGS["PromptKeyFilter"] = "Prompt Key Filter"

    # Register API endpoint for templates (single source of truth)
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get("/api/z_image_templates")
        async def get_z_image_templates(request):
            """API endpoint for Z-Image templates - JS fetches from here."""
            templates = get_templates()
            return web.json_response(templates)

        print("[QwenImageWanBridge] Loaded Z-Image encoder nodes (3 nodes)")
        print("[QwenImageWanBridge] API: /api/z_image_templates endpoint registered")
    except Exception as api_err:
        print(f"[QwenImageWanBridge] Loaded Z-Image nodes (API endpoint failed: {api_err})")

except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Z-Image encoder nodes: {e}")

# ============================================================================
# EXPERIMENTAL ANALYSIS NODES
# ============================================================================

# Template Influence Analyzer - tests if system prompts affect embeddings
try:
    from .nodes.template_influence_analyzer import TemplateInfluenceAnalyzer

    NODE_CLASS_MAPPINGS["TemplateInfluenceAnalyzer"] = TemplateInfluenceAnalyzer
    NODE_DISPLAY_NAME_MAPPINGS["TemplateInfluenceAnalyzer"] = "Template Influence Analyzer"

    print("[QwenImageWanBridge] Loaded Template Influence Analyzer (experimental)")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Template Influence Analyzer: {e}")

# ============================================================================
# DEBUG TRACING - Opt-in via environment variable
# ============================================================================

import os
if os.environ.get('QWEN_ENABLE_DEBUG_PATCHES', '').lower() in ('true', '1', 'yes'):
    try:
        from .nodes import debug_patch
        debug_patch.apply_debug_patches()
        print("[QwenImageWanBridge] Debug patches applied (requested via QWEN_ENABLE_DEBUG_PATCHES)")
        print("[QwenImageWanBridge] Set QWEN_DEBUG_VERBOSE=true for verbose tracing output")
    except Exception as e:
        print(f"[QwenImageWanBridge] Debug patches failed: {e}")
