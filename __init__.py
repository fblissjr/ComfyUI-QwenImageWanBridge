"""
ComfyUI-QwenImageWanBridge
Custom nodes for bridging Qwen-Image and WAN video models
"""

# Direct imports like shrug-prompter
from .nodes.diagnostic import VAEDiagnosticNode, LatentAnalyzer
from .nodes.bridge import QwenWANBridge, LatentMixer
from .nodes.latent_explorer import (
    SemanticDirectionFinder, LatentWalk, LatentPCA, StyleContentSeparator
)

NODE_CLASS_MAPPINGS = {
    # Diagnostic nodes
    "VAEDiagnosticNode": VAEDiagnosticNode,
    "LatentAnalyzer": LatentAnalyzer,

    # Bridge nodes
    "QwenWANBridge": QwenWANBridge,
    "LatentMixer": LatentMixer,
    
    # Explorer nodes
    "SemanticDirectionFinder": SemanticDirectionFinder,
    "LatentWalk": LatentWalk,
    "LatentPCA": LatentPCA,
    "StyleContentSeparator": StyleContentSeparator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Diagnostic nodes
    "VAEDiagnosticNode": "VAE Diagnostic",
    "LatentAnalyzer": "Latent Analyzer",

    # Bridge nodes
    "QwenWANBridge": "Qwen-WAN Bridge",
    "LatentMixer": "Latent Mixer",
    
    # Explorer nodes
    "SemanticDirectionFinder": "Semantic Direction Finder",
    "LatentWalk": "Latent Walk",
    "LatentPCA": "Latent PCA",
    "StyleContentSeparator": "Style/Content Separator",
}

print(f"[QwenImageWanBridge] Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
