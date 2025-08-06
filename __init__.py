"""
ComfyUI-QwenWANBridge
Custom nodes for bridging Qwen-Image and WAN video models
"""

import os
import sys
import traceback

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import nodes with error handling
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def safe_import(module_name, class_name, display_name):
    """Safely import a node class"""
    try:
        module = __import__(module_name, fromlist=[class_name])
        node_class = getattr(module, class_name)
        NODE_CLASS_MAPPINGS[class_name] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name
        print(f"✓ Loaded: {display_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to load {display_name}: {e}")
        return False

# Import basic nodes first
safe_import("nodes.diagnostic", "VAEDiagnosticNode", "VAE Diagnostic")
safe_import("nodes.diagnostic", "SimplifiedVAETest", "Simple VAE Test")
safe_import("nodes.bridge", "QwenWANBridge", "Qwen-WAN Bridge")
safe_import("nodes.bridge", "AlwaysUseWANVAE", "Always Use WAN VAE")

# Import advanced nodes if dependencies are available
try:
    import requests
    safe_import("nodes.vlm", "ShrugPrompterNode", "Shrug Prompter VLM")
except ImportError:
    print("⚠ Shrug Prompter disabled (install requests)")

try:
    import safetensors
    safe_import("nodes.lora", "LoRATransferNode", "LoRA Transfer")
except ImportError:
    print("LoRA Transfer disabled (install safetensors)")

# Print summary
print(f"\nQwen-WAN Bridge: Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
print("Key insight: Always use WAN VAE for both models to avoid bizarre frames!")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
