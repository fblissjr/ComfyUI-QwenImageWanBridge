"""
Debug nodes for inspecting latents and understanding what's happening
"""

import torch
import numpy as np

class QwenWANLatentDebug:
    """
    Debug node to inspect latent tensors and show detailed statistics
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "label": ("STRING", {"default": "Latent"}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "debug_info")
    FUNCTION = "debug"
    CATEGORY = "QwenWANBridge/Debug"
    
    def debug(self, latent, label):
        info = []
        info.append(f"=== {label} Debug Info ===")
        
        samples = latent["samples"]
        info.append(f"Shape: {samples.shape}")
        info.append(f"Device: {samples.device}")
        info.append(f"Dtype: {samples.dtype}")
        
        # Statistics
        info.append("\n--- Statistics ---")
        info.append(f"Min: {samples.min().item():.6f}")
        info.append(f"Max: {samples.max().item():.6f}")
        info.append(f"Mean: {samples.mean().item():.6f}")
        info.append(f"Std: {samples.std().item():.6f}")
        
        # Check for issues
        info.append("\n--- Checks ---")
        
        # Check if all zeros
        if torch.allclose(samples, torch.zeros_like(samples)):
            info.append("WARNING: Latent is all zeros!")
        else:
            info.append("OK: Latent contains non-zero values")
        
        # Check if all same value
        if samples.std().item() < 0.0001:
            info.append(f"WARNING: Latent has no variation (std={samples.std().item():.6f})")
        else:
            info.append(f"OK: Latent has variation (std={samples.std().item():.4f})")
        
        # Check for NaN or Inf
        if torch.isnan(samples).any():
            info.append("ERROR: Latent contains NaN values!")
        elif torch.isinf(samples).any():
            info.append("ERROR: Latent contains Inf values!")
        else:
            info.append("OK: No NaN or Inf values")
        
        # Check value range
        if samples.min() < -10 or samples.max() > 10:
            info.append(f"WARNING: Values outside normal range [-10, 10]")
        else:
            info.append("OK: Values in normal range")
        
        # Shape analysis
        info.append("\n--- Shape Analysis ---")
        if len(samples.shape) == 4:
            B, C, H, W = samples.shape
            info.append(f"Batch: {B}, Channels: {C}, Height: {H}, Width: {W}")
            info.append("Type: Single frame latent")
        elif len(samples.shape) == 5:
            B, C, T, H, W = samples.shape
            info.append(f"Batch: {B}, Channels: {C}, Frames: {T}, Height: {H}, Width: {W}")
            info.append("Type: Video latent")
            
            # Check temporal consistency
            if T > 1:
                frame_diff = samples[:, :, 1:] - samples[:, :, :-1]
                temporal_change = frame_diff.abs().mean().item()
                info.append(f"Temporal change: {temporal_change:.6f}")
                if temporal_change < 0.0001:
                    info.append("WARNING: No temporal variation (static video)")
        
        # Channel analysis
        info.append("\n--- Channel Analysis ---")
        C = samples.shape[1]
        if C == 16:
            info.append("Channels: 16 (Compatible with WAN 2.1 / Qwen)")
        elif C == 48:
            info.append("Channels: 48 (WAN 2.2)")
        elif C == 4:
            info.append("Channels: 4 (Standard SD latent)")
        else:
            info.append(f"Channels: {C} (Unusual)")
        
        # Per-channel stats
        for i in range(min(4, C)):  # First 4 channels
            channel_mean = samples[:, i].mean().item()
            channel_std = samples[:, i].std().item()
            info.append(f"  Channel {i}: mean={channel_mean:.4f}, std={channel_std:.4f}")
        
        # Sample some actual values
        info.append("\n--- Sample Values ---")
        info.append("First 5 values (flattened):")
        first_values = samples.flatten()[:5].cpu().numpy()
        info.append(f"  {first_values}")
        
        debug_text = "\n".join(info)
        print(debug_text)  # Also print to console
        
        return (latent, debug_text)


class QwenWANConditioningDebug:
    """
    Debug node to inspect conditioning dictionaries
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "label": ("STRING", {"default": "Conditioning"}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "debug_info")
    FUNCTION = "debug"
    CATEGORY = "QwenWANBridge/Debug"
    
    def debug(self, conditioning, label):
        info = []
        info.append(f"=== {label} Debug Info ===")
        
        if isinstance(conditioning, list):
            info.append(f"Conditioning list length: {len(conditioning)}")
            
            for i, cond in enumerate(conditioning):
                info.append(f"\n--- Item {i} ---")
                
                if isinstance(cond, tuple) and len(cond) >= 2:
                    embeddings, metadata = cond[0], cond[1]
                    
                    # Check embeddings
                    if isinstance(embeddings, torch.Tensor):
                        info.append(f"Embeddings shape: {embeddings.shape}")
                        info.append(f"Embeddings mean: {embeddings.mean().item():.4f}")
                        info.append(f"Embeddings std: {embeddings.std().item():.4f}")
                        
                        # Check dimensions
                        if embeddings.shape[-1] == 768:
                            info.append("Type: CLIP embeddings (768 dim)")
                        elif embeddings.shape[-1] == 4096:
                            info.append("Type: UMT5-XXL embeddings (4096 dim)")
                        else:
                            info.append(f"Type: Unknown ({embeddings.shape[-1]} dim)")
                    
                    # Check metadata
                    if isinstance(metadata, dict):
                        info.append(f"Metadata keys: {list(metadata.keys())}")
                        
                        # Check for I2V conditioning
                        if "concat_latent_image" in metadata:
                            concat = metadata["concat_latent_image"]
                            info.append(f"  concat_latent_image shape: {concat.shape}")
                            info.append(f"  concat_latent_image mean: {concat.mean().item():.4f}")
                            
                            if torch.allclose(concat, torch.zeros_like(concat)):
                                info.append("  WARNING: concat_latent_image is all zeros!")
                        
                        if "concat_mask" in metadata:
                            mask = metadata["concat_mask"]
                            info.append(f"  concat_mask shape: {mask.shape}")
                            info.append(f"  concat_mask mean: {mask.mean().item():.4f}")
                        
                        if "clip_vision_output" in metadata:
                            info.append("  Has CLIP vision output")
        else:
            info.append(f"Conditioning type: {type(conditioning)}")
        
        debug_text = "\n".join(info)
        print(debug_text)  # Also print to console
        
        return (conditioning, debug_text)


class QwenWANCompareLatents:
    """
    Compare two latents to see differences
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "label_a": ("STRING", {"default": "Latent A"}),
                "label_b": ("STRING", {"default": "Latent B"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("comparison",)
    FUNCTION = "compare"
    CATEGORY = "QwenWANBridge/Debug"
    
    def compare(self, latent_a, latent_b, label_a, label_b):
        info = []
        info.append(f"=== Comparing {label_a} vs {label_b} ===")
        
        a = latent_a["samples"]
        b = latent_b["samples"]
        
        # Shape comparison
        info.append("\n--- Shapes ---")
        info.append(f"{label_a}: {a.shape}")
        info.append(f"{label_b}: {b.shape}")
        
        if a.shape != b.shape:
            info.append("WARNING: Different shapes!")
            
            # Try to compare what we can
            if a.shape[1] != b.shape[1]:
                info.append(f"Different channels: {a.shape[1]} vs {b.shape[1]}")
            
            # Can't do element-wise comparison
            info.append("Cannot do detailed comparison due to shape mismatch")
        else:
            # Statistics comparison
            info.append("\n--- Statistics ---")
            info.append(f"{label_a} - Mean: {a.mean().item():.4f}, Std: {a.std().item():.4f}")
            info.append(f"{label_b} - Mean: {b.mean().item():.4f}, Std: {b.std().item():.4f}")
            
            # Difference metrics
            info.append("\n--- Differences ---")
            diff = (a - b).abs()
            info.append(f"Mean absolute difference: {diff.mean().item():.6f}")
            info.append(f"Max absolute difference: {diff.max().item():.6f}")
            
            # Correlation
            a_flat = a.flatten()
            b_flat = b.flatten()
            if len(a_flat) > 0:
                correlation = torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()
                info.append(f"Correlation: {correlation:.4f}")
            
            # Check if identical
            if torch.allclose(a, b, rtol=1e-5):
                info.append("Latents are nearly identical!")
            elif torch.allclose(a, b, rtol=1e-3):
                info.append("Latents are very similar")
            elif torch.allclose(a, b, rtol=1e-1):
                info.append("Latents are somewhat similar")
            else:
                info.append("Latents are significantly different")
        
        comparison = "\n".join(info)
        print(comparison)
        
        return (comparison,)