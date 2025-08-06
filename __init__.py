"""
ComfyUI-QwenImageWanBridge
Custom nodes for bridging Qwen-Image and WAN video models
"""

# Direct imports like shrug-prompter
from .nodes.diagnostic import VAEDiagnosticNode, SimplifiedVAETest
from .nodes.bridge import QwenWANBridge, AlwaysUseWANVAE, LatentMixer

NODE_CLASS_MAPPINGS = {
    # Diagnostic nodes
    "VAEDiagnosticNode": VAEDiagnosticNode,
    "SimplifiedVAETest": SimplifiedVAETest,

    # Bridge nodes
    "QwenWANBridge": QwenWANBridge,
    "AlwaysUseWANVAE": AlwaysUseWANVAE,
    "LatentMixer": LatentMixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Diagnostic nodes
    "VAEDiagnosticNode": "VAE Diagnostic",
    "SimplifiedVAETest": "Simple VAE Test",

    # Bridge nodes
    "QwenWANBridge": "Qwen-WAN Bridge",
    "AlwaysUseWANVAE": "Always Use WAN VAE",
    "LatentMixer": "Latent Mixer",
}

print(f"[QwenImageWanBridge] Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
print("[QwenImageWanBridge] Key insight: Always use WAN VAE to avoid bizarre frames")
