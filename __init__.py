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

# Note: Experimental multi-frame nodes have been archived
# Use Multi-Reference Handler with "index" mode for multi-frame support

# ============================================================================
# PATCHES - Apply monkey patches for multi-frame support
# ============================================================================

try:
    from .nodes import qwen_encoder_patch
    print("[QwenImageWanBridge] Applied encoder patches for multi-frame support")
except Exception as e:
    print(f"[QwenImageWanBridge] Could not apply patches: {e}")

# ============================================================================
# NATIVE NODES - Direct Qwen2.5-VL bypassing ComfyUI CLIP limitations  
# ============================================================================

try:
    from .nodes.native.qwen_native_loader import QwenNativeLoader
    from .nodes.native.qwen_native_encoder import QwenNativeEncoder
    from .nodes.native.qwen_context_processor import QwenContextProcessor
    from .nodes.native.qwen_validation_tools import QwenValidationTools

    NODE_CLASS_MAPPINGS["QwenNativeLoader"] = QwenNativeLoader
    NODE_DISPLAY_NAME_MAPPINGS["QwenNativeLoader"] = "Qwen Native Loader"

    NODE_CLASS_MAPPINGS["QwenNativeEncoder"] = QwenNativeEncoder
    NODE_DISPLAY_NAME_MAPPINGS["QwenNativeEncoder"] = "Qwen Native Encoder"

    NODE_CLASS_MAPPINGS["QwenContextProcessor"] = QwenContextProcessor
    NODE_DISPLAY_NAME_MAPPINGS["QwenContextProcessor"] = "Qwen Context Processor"

    NODE_CLASS_MAPPINGS["QwenValidationTools"] = QwenValidationTools
    NODE_DISPLAY_NAME_MAPPINGS["QwenValidationTools"] = "Qwen Validation Tools"

    print("[QwenImageWanBridge] Loaded Native Qwen nodes (4 nodes)")
    print("[QwenImageWanBridge] FIXES: Template dropping, Vision duplication, Processor path, Context images")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Native nodes: {e}")

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[QwenImageWanBridge] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

# ============================================================================
# STANDALONE ENGINE - Qwen Image Engine based on DiffSynth-Engine architecture
# ============================================================================

try:
    from .engine.nodes.engine_loader import QwenEngineLoader
    from .engine.nodes.engine_encoder import QwenEngineEncoder  
    from .engine.nodes.engine_sampler import QwenEngineSampler

    NODE_CLASS_MAPPINGS["QwenEngineLoader"] = QwenEngineLoader
    NODE_DISPLAY_NAME_MAPPINGS["QwenEngineLoader"] = "Qwen Engine Loader"

    NODE_CLASS_MAPPINGS["QwenEngineEncoder"] = QwenEngineEncoder
    NODE_DISPLAY_NAME_MAPPINGS["QwenEngineEncoder"] = "Qwen Engine Encoder"

    NODE_CLASS_MAPPINGS["QwenEngineSampler"] = QwenEngineSampler
    NODE_DISPLAY_NAME_MAPPINGS["QwenEngineSampler"] = "Qwen Engine Sampler"

    print("[QwenImageWanBridge] Loaded Standalone Qwen Engine nodes (3 nodes)")
    print("[QwenImageWanBridge] Based on DiffSynth-Engine architecture with proper reference latent handling")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Engine nodes: {e}")

# ============================================================================
# AUTO DEBUG TRACING - Apply patches automatically
# ============================================================================

try:
    from .nodes import debug_patch
    debug_patch.apply_debug_patches()
    print("[QwenImageWanBridge] ✅ Debug patches applied - end-to-end tracing active")
    print("[QwenImageWanBridge] Trace shows: Model Load → Conditioning → Sampler → Forward Pass")
except Exception as e:
    print(f"[QwenImageWanBridge] ❌ Debug patches failed: {e}")
