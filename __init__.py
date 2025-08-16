"""
ComfyUI-QwenImageWanBridge
Direct latent bridge from Qwen-Image to WAN 2.2 video
No VAE decode/encode needed - pure latent space operation
"""

from .nodes.qwen_wan_latent_bridge import QwenImageToWANLatentBridge

NODE_CLASS_MAPPINGS = {
    "QwenImageToWANLatentBridge": QwenImageToWANLatentBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageToWANLatentBridge": "Qwen→WAN Latent Bridge",
}

print("[QwenImageWanBridge] One node: Direct latent transfer from Qwen-Image to WAN 2.2")