"""
ComfyUI-QwenImageWanBridge
Direct latent bridge from Qwen-Image to WAN 2.2 video generation
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ============================================================================
# PRODUCTION-READY NODES
# ============================================================================

try:
    from .nodes.qwen_wan_pure_bridge import QwenWANPureBridge, QwenWANMappingAnalyzer
    NODE_CLASS_MAPPINGS["QwenWANPureBridge"] = QwenWANPureBridge
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANPureBridge"] = "Qwen→WAN Pure Bridge"
    
    NODE_CLASS_MAPPINGS["QwenWANMappingAnalyzer"] = QwenWANMappingAnalyzer
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANMappingAnalyzer"] = "Qwen→WAN Mapping Analyzer"
    
    print("[QwenImageWanBridge] Loaded Pure Bridge nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Pure Bridge: {e}")

try:
    from .nodes.qwen_wan_semantic_bridge import QwenWANSemanticBridge, QwenWANDimensionHelper
    NODE_CLASS_MAPPINGS["QwenWANSemanticBridge"] = QwenWANSemanticBridge
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANSemanticBridge"] = "Qwen→WAN Semantic Bridge"
    
    NODE_CLASS_MAPPINGS["QwenWANDimensionHelper"] = QwenWANDimensionHelper
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANDimensionHelper"] = "Qwen→WAN Dimension Helper"
    
    print("[QwenImageWanBridge] Loaded Semantic Bridge nodes")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load Semantic Bridge: {e}")

# ============================================================================
# RESEARCH/TESTING NODES (Optional - uncomment to enable)
# ============================================================================

LOAD_RESEARCH_NODES = False  # Set to True to load research nodes

if LOAD_RESEARCH_NODES:
    try:
        from .nodes.research.qwen_wan_parameter_sweep import (
            QwenWANParameterSweep, 
            QwenWANBestSettings,
            QwenWANTestRunner
        )
        NODE_CLASS_MAPPINGS["QwenWANParameterSweep"] = QwenWANParameterSweep
        NODE_DISPLAY_NAME_MAPPINGS["QwenWANParameterSweep"] = "[TEST] Parameter Sweep"
        
        NODE_CLASS_MAPPINGS["QwenWANBestSettings"] = QwenWANBestSettings
        NODE_DISPLAY_NAME_MAPPINGS["QwenWANBestSettings"] = "[TEST] Best Settings"
        
        NODE_CLASS_MAPPINGS["QwenWANTestRunner"] = QwenWANTestRunner
        NODE_DISPLAY_NAME_MAPPINGS["QwenWANTestRunner"] = "[TEST] Test Runner"
        
        print("[QwenImageWanBridge] Loaded research/testing nodes")
    except Exception as e:
        print(f"[QwenImageWanBridge] Failed to load research nodes: {e}")

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[QwenImageWanBridge] Registered {len(NODE_CLASS_MAPPINGS)} nodes")
print("[QwenImageWanBridge] Production nodes ready. Set LOAD_RESEARCH_NODES=True in __init__.py to enable testing nodes.")