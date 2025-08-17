"""
ComfyUI-QwenImageWanBridge
Direct latent bridge from Qwen-Image to WAN 2.2 video
No VAE decode/encode needed - pure latent space operation
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes.qwen_wan_bridge import QwenWANBridge
    NODE_CLASS_MAPPINGS["QwenWANBridge"] = QwenWANBridge
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANBridge"] = "Qwen→WAN Bridge"
    print("[QwenImageWanBridge] Loaded QwenWANBridge")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load QwenWANBridge: {e}")

try:
    from .nodes.qwen_wan_bridge_v2 import QwenWANBridgeV2
    NODE_CLASS_MAPPINGS["QwenWANBridgeV2"] = QwenWANBridgeV2
    NODE_DISPLAY_NAME_MAPPINGS["QwenWANBridgeV2"] = "Qwen→WAN Bridge V2 (Exact)"
    print("[QwenImageWanBridge] Loaded QwenWANBridgeV2")
except Exception as e:
    print(f"[QwenImageWanBridge] Failed to load QwenWANBridgeV2: {e}")

# Required exports for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[QwenImageWanBridge] Registered {len(NODE_CLASS_MAPPINGS)} nodes")