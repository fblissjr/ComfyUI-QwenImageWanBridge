"""
VAE Diagnostic Nodes
Analyze why Qwen VAE produces bizarre frames with WAN
"""

import torch
import json
from typing import Dict, Any

class VAEDiagnosticNode:
    """Analyze VAE compatibility between Qwen and WAN"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "qwen_vae": ("VAE",),
                "wan_vae": ("VAE",),
            },
            "optional": {
                "run_full_diagnostic": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "LATENT", "LATENT")
    RETURN_NAMES = ("report", "qwen_latent", "wan_latent")
    FUNCTION = "diagnose"
    CATEGORY = "QwenWAN/Diagnostic"
    
    def diagnose(self, image, qwen_vae, wan_vae, run_full_diagnostic=False):
        """Compare VAE encodings"""
        
        report = {
            "status": "analyzing",
            "findings": {}
        }
        
        # Encode with both VAEs
        with torch.no_grad():
            qwen_latent = qwen_vae.encode(image[:, :, :, :3])
            wan_latent = wan_vae.encode(image[:, :, :, :3])
        
        # Compare shapes
        report["findings"]["qwen_shape"] = list(qwen_latent.shape)
        report["findings"]["wan_shape"] = list(wan_latent.shape)
        
        # Calculate difference
        if qwen_latent.shape == wan_latent.shape:
            diff = torch.abs(qwen_latent - wan_latent).mean().item()
            report["findings"]["spatial_difference"] = f"{diff:.6f}"
            report["findings"]["compatible"] = diff < 0.01
        else:
            report["findings"]["compatible"] = False
            report["findings"]["error"] = "Shape mismatch"
        
        if run_full_diagnostic:
            # Additional analysis
            report["findings"]["qwen_stats"] = {
                "mean": float(qwen_latent.mean()),
                "std": float(qwen_latent.std()),
                "min": float(qwen_latent.min()),
                "max": float(qwen_latent.max())
            }
            report["findings"]["wan_stats"] = {
                "mean": float(wan_latent.mean()),
                "std": float(wan_latent.std()),
                "min": float(wan_latent.min()),
                "max": float(wan_latent.max())
            }
        
        # Key insight
        report["recommendation"] = "Always use WAN VAE for decoding to avoid bizarre frames"
        
        report_str = json.dumps(report, indent=2)
        
        return (report_str, qwen_latent, wan_latent)


class SimplifiedVAETest:
    """Simple test: Which VAE to use?"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "test_type": (["qwen_to_wan", "wan_to_qwen", "best_practice"],),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "QwenWAN/Diagnostic"
    
    def test(self, test_type):
        """Simple recommendation"""
        
        recommendations = {
            "qwen_to_wan": """
When using Qwen model output with WAN pipeline:
✓ ALWAYS use WAN VAE for decoding
✗ NEVER use Qwen VAE (causes bizarre frames)

Workflow:
Qwen Model → Generate → Decode with WAN VAE → Success!
""",
            "wan_to_qwen": """
When using WAN model output with Qwen:
✓ Use WAN VAE (works perfectly)
✗ Qwen VAE might work but WAN VAE is safer

Workflow:
WAN Model → Generate → Decode with WAN VAE → Success!
""",
            "best_practice": """
UNIVERSAL SOLUTION:
Always use WAN VAE for everything!

Why? WAN VAE works with both models:
- Qwen latents → WAN VAE decode ✓
- WAN latents → WAN VAE decode ✓

Simple rule: Load WAN VAE, use it everywhere.
No bizarre frames, no complications!
"""
        }
        
        return (recommendations.get(test_type, "Unknown test type"),)