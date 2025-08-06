"""
VAE Diagnostic Nodes
Analyze why Qwen VAE produces bizarre frames with WAN
"""

import torch
import json
from typing import Dict, Any

class VAEDiagnosticNode:
    """Compare how different VAEs decode the same latent"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae1": ("VAE",),
                "vae2": ("VAE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("vae1_decoded", "vae2_decoded", "report")
    FUNCTION = "diagnose"
    CATEGORY = "QwenWAN/Diagnostic"
    
    def diagnose(self, latent, vae1, vae2):
        """Decode same latent with both VAEs to see differences"""
        
        report = {
            "test": "VAE Decode Comparison",
            "findings": {}
        }
        
        # Get latent samples
        samples = latent["samples"] if isinstance(latent, dict) else latent
        
        # Decode with both VAEs using ComfyUI's method
        with torch.no_grad():
            # Use the ComfyUI VAE decode method which returns properly formatted images
            image1 = vae1.decode(samples)
            image2 = vae2.decode(samples)
        
        # ComfyUI VAEs should return images in (B, H, W, C) format
        # but some custom VAEs might return (B, C, H, W)
        if image1.dim() == 4 and image1.shape[-1] > 4:
            # Already in (B, H, W, C) format - good!
            pass
        elif image1.dim() == 4 and image1.shape[1] <= 4:
            # In (B, C, H, W) format - need to permute
            image1 = image1.permute(0, 2, 3, 1)
        
        if image2.dim() == 4 and image2.shape[-1] > 4:
            # Already in (B, H, W, C) format - good!
            pass
        elif image2.dim() == 4 and image2.shape[1] <= 4:
            # In (B, C, H, W) format - need to permute
            image2 = image2.permute(0, 2, 3, 1)
        
        # Calculate difference
        if image1.shape == image2.shape:
            pixel_diff = torch.abs(image1 - image2).mean().item()
            report["findings"]["pixel_difference"] = f"{pixel_diff:.4f}"
            report["findings"]["similarity"] = f"{(1 - pixel_diff) * 100:.2f}%"
            
            # Check for bizarre artifacts
            if pixel_diff > 0.1:
                report["findings"]["warning"] = "Significant differences detected - may cause bizarre frames"
            else:
                report["findings"]["status"] = "VAEs are compatible"
        else:
            report["findings"]["error"] = f"Shape mismatch: {image1.shape} vs {image2.shape}"
        
        report["recommendation"] = "Use WAN VAE for both Qwen and WAN models to avoid issues"
        
        report_str = json.dumps(report, indent=2)
        
        return (image1, image2, report_str)


class LatentAnalyzer:
    """Analyze latent characteristics"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze"
    CATEGORY = "QwenWAN/Diagnostic"
    
    def analyze(self, latent):
        """Analyze latent tensor properties"""
        
        samples = latent["samples"] if isinstance(latent, dict) else latent
        
        analysis = {
            "shape": list(samples.shape),
            "dtype": str(samples.dtype),
            "device": str(samples.device),
            "stats": {
                "mean": float(samples.mean()),
                "std": float(samples.std()),
                "min": float(samples.min()),
                "max": float(samples.max())
            },
            "dimensions": {
                "batch": samples.shape[0] if len(samples.shape) > 0 else None,
                "channels": samples.shape[1] if len(samples.shape) > 1 else None,
                "height": samples.shape[-2] if len(samples.shape) > 2 else None,
                "width": samples.shape[-1] if len(samples.shape) > 1 else None,
            }
        }
        
        if len(samples.shape) == 5:
            analysis["dimensions"]["frames"] = samples.shape[2]
            analysis["type"] = "video_latent"
        else:
            analysis["type"] = "image_latent"
        
        return (json.dumps(analysis, indent=2),)