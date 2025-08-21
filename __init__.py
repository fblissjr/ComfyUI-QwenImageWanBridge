"""
ComfyUI-QwenImageWanBridge
Direct latent bridge from Qwen-Image to WAN video generation

STATUS: Partially functional - quality degradation due to VAE differences
RECOMMENDATION: Use WAN 2.1 (16ch) instead of WAN 2.2 (48ch) for better compatibility
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ============================================================================
# PRODUCTION NODES - Native ComfyUI
# ============================================================================

# Note: Research nodes moved to nodes/research/ folder
# Only loading production-ready unified nodes

try:
    from .nodes.qwen_wan_i2v_bridge import QwenWANI2VBridge, QwenToImage, QwenWANI2VDirect
    NODE_CLASS_MAPPINGS["QwenWANI2VBridge"] = QwenWANI2VBridge
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANI2VBridge"] = "Qwen WAN I2V Conditioning"
    
    NODE_CLASS_MAPPINGS["QwenToImage"] = QwenToImage
    NODE_DISPLAY_NAME_MAPPINGS["QwenToImage"] = "Qwen Latent to Image"
    
    NODE_CLASS_MAPPINGS["QwenWANI2VDirect"] = QwenWANI2VDirect
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANI2VDirect"] = "Qwen WAN I2V Direct"
    
    print("[QwenImageWanBridge] Loaded I2V specific nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load I2V nodes: {e}")

try:
    from .nodes.qwen_wan_unified_i2v import QwenWANUnifiedI2V
    NODE_CLASS_MAPPINGS["QwenWANUnifiedI2V"] = QwenWANUnifiedI2V
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANUnifiedI2V"] = "Qwen WAN Unified I2V"
    print("[QwenImageWanBridge] Loaded Unified I2V node")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Unified I2V: {e}")

try:
    from .nodes.qwen_wan_unified_t2v import QwenWANUnifiedT2V
    NODE_CLASS_MAPPINGS["QwenWANUnifiedT2V"] = QwenWANUnifiedT2V
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANUnifiedT2V"] = "Qwen WAN Unified T2V"
    print("[QwenImageWanBridge] Loaded Unified T2V node")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Unified T2V: {e}")

# ============================================================================
# ARCHIVED NODES (Broken/Superseded - kept for reference)
# ============================================================================

LOAD_ARCHIVED_NODES = False  # Set to True to load archived nodes

if LOAD_ARCHIVED_NODES:
    try:
        from .nodes.archive.qwen_wan_pure_bridge import QwenWANPureBridge, QwenWANMappingAnalyzer
        NODE_CLASS_MAPPINGS["QwenWANPureBridge"] = QwenWANPureBridge
        NODE_DISPLAY_NAME_MAPPINGS["QwenWANPureBridge"] = "[ARCHIVED] Pure Bridge"
        NODE_CLASS_MAPPINGS["QwenWANMappingAnalyzer"] = QwenWANMappingAnalyzer
        NODE_DISPLAY_NAME_MAPPINGS["QwenWANMappingAnalyzer"] = "[ARCHIVED] Mapping Analyzer"
        print("[QwenImageWanBridge] Loaded archived nodes")
    except Exception as e:
        print(f"[QwenImageWanBridge] Failed to load archived nodes: {e}")

# ============================================================================
# RESEARCH NODES (Experimental - for testing only)
# ============================================================================

LOAD_RESEARCH_NODES = False  # Set to True to load research nodes

if LOAD_RESEARCH_NODES:
    try:
        from .nodes.research.qwen_wan_parameter_sweep import (
            QwenWANParameterSweep, 
            QwenWANBestSettings,
        )
        NODE_CLASS_MAPPINGS["QwenWANParameterSweep"] = QwenWANParameterSweep
        NODE_DISPLAY_NAME_MAPPINGS["QwenWANParameterSweep"] = "[RESEARCH] Parameter Sweep"
        NODE_CLASS_MAPPINGS["QwenWANBestSettings"] = QwenWANBestSettings
        NODE_DISPLAY_NAME_MAPPINGS["QwenWANBestSettings"] = "[RESEARCH] Best Settings"
        print("[QwenImageWanBridge] Loaded research nodes")
    except Exception as e:
        print(f"[QwenImageWanBridge] Failed to load research nodes: {e}")

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[QwenImageWanBridge] {len(NODE_CLASS_MAPPINGS)} active nodes")
print("[QwenImageWanBridge] Key discovery: WAN 2.1 (16ch) compatible, WAN 2.2 (48ch) needs channel adapter")
print("[QwenImageWanBridge] Set LOAD_ARCHIVED_NODES=True or LOAD_RESEARCH_NODES=True to access experimental nodes")