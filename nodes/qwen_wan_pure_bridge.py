"""
Pure minimal bridge - NO noise, NO normalization, NO modifications
Just structural adaptation for WAN's expected format
Let the sampler handle ALL denoising decisions
"""

import torch
import torch.nn.functional as F

class QwenWANPureBridge:
    """
    Absolute minimal bridge - only handles structural requirements
    NO noise addition, NO normalization, NO modifications
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 9, "min": 1, "max": 1024, "step": 4}),
                "mode": (["i2v", "v2v", "both"], {"default": "both"}),
            }
        }
    
    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "LATENT", "STRING")
    RETURN_NAMES = ("image_embeds", "samples", "info")
    FUNCTION = "bridge"
    CATEGORY = "QwenWANBridge/Pure"
    
    def bridge(self, qwen_latent, width, height, num_frames, mode):
        
        info = []
        info.append("Pure Bridge - No Modifications")
        info.append("="*40)
        
        # Extract Qwen latent
        qwen = qwen_latent["samples"]
        
        # Handle batch dimension
        if len(qwen.shape) == 5:
            # (B, C, F, H, W)
            qwen = qwen[0, :, 0, :, :]  # Take first batch, first frame
        elif len(qwen.shape) == 4:
            # (B, C, H, W)
            qwen = qwen[0]  # Take first batch
        # Now (C, H, W)
        
        C, H_orig, W_orig = qwen.shape
        device = qwen.device
        dtype = qwen.dtype
        
        info.append(f"Input: {H_orig}x{W_orig}, {C} channels")
        
        # Calculate target dimensions
        lat_h = height // 8
        lat_w = width // 8
        
        # Resize if needed (bilinear, no tricks)
        if (H_orig, W_orig) != (lat_h, lat_w):
            qwen_resized = F.interpolate(
                qwen.unsqueeze(0),
                size=(lat_h, lat_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            info.append(f"Resized to: {lat_h}x{lat_w}")
        else:
            qwen_resized = qwen
            info.append("No resize needed")
        
        # Frame alignment for WAN
        num_frames_aligned = ((num_frames - 1) // 4) * 4 + 1
        temporal_frames = (num_frames_aligned - 1) // 4 + 1
        
        info.append(f"Frames: {num_frames} → {num_frames_aligned} (aligned)")
        info.append(f"Temporal: {temporal_frames}")
        
        # Create temporal tensor - PURE, no modifications
        latent_frames = torch.zeros(C, temporal_frames, lat_h, lat_w, 
                                   device=device, dtype=dtype)
        
        # Place Qwen in first frame EXACTLY as is
        latent_frames[:, 0] = qwen_resized
        
        # Rest remain zeros (let sampler decide what to do)
        info.append(f"\nFrame 0: Qwen latent (unmodified)")
        info.append(f"Frames 1-{temporal_frames-1}: zeros")
        
        # Create mask - standard I2V mask
        mask = torch.zeros(1, num_frames_aligned, lat_h, lat_w, device=device)
        mask[:, 0] = 1.0  # First frame has content
        
        # Reshape mask for WAN format
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)
        
        # Ensure correct size
        frames_needed = temporal_frames * 4
        if mask.shape[1] < frames_needed:
            padding = torch.zeros(1, frames_needed - mask.shape[1], lat_h, lat_w, device=device)
            mask = torch.cat([mask, padding], dim=1)
        elif mask.shape[1] > frames_needed:
            mask = mask[:, :frames_needed]
        
        mask = mask.view(1, temporal_frames, 4, lat_h, lat_w)
        mask = mask.movedim(1, 2)[0]  # (4, T, H, W)
        
        # Statistics
        info.append(f"\nLatent stats:")
        info.append(f"  Min: {qwen_resized.min():.3f}")
        info.append(f"  Max: {qwen_resized.max():.3f}")
        info.append(f"  Mean: {qwen_resized.mean():.3f}")
        info.append(f"  Std: {qwen_resized.std():.3f}")
        
        # Create outputs based on mode
        image_embeds = None
        samples = None
        
        if mode in ["i2v", "both"]:
            # I2V structure
            image_embeds = {
                "image_embeds": latent_frames,
                "mask": mask,
                "num_frames": num_frames_aligned,
                "lat_h": lat_h,
                "lat_w": lat_w,
                "target_shape": (C, temporal_frames, lat_h, lat_w),
                "has_ref": False,
                "fun_or_fl2v_model": False,
                # Minimal metadata
                "clip_context": None,
                "negative_clip_context": None,
                "control_embeds": None,
                "add_cond_latents": None,
                "end_image": None,
                "max_seq_len": lat_h * lat_w // 4 * temporal_frames,
            }
            info.append("\nI2V embeds created")
        
        if mode in ["v2v", "both"]:
            # V2V samples - add batch dimension back
            samples = {"samples": latent_frames.unsqueeze(0)}
            info.append("V2V samples created (with batch dim)")
        
        info.append(f"\nMode: {mode}")
        info.append("\nNOTE: Use sampler's denoise parameter to control")
        info.append("how much the output differs from input.")
        info.append("Lower denoise = more preservation")
        info.append("Higher denoise = more generation")
        
        return (image_embeds or {}, samples or {"samples": torch.zeros(1, C, temporal_frames, lat_h, lat_w)}, "\n".join(info))


class QwenWANMappingAnalyzer:
    """
    Analyze exactly how latents should be mapped between models
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "analysis_type": ([
                    "distribution",
                    "channel_analysis", 
                    "spatial_patterns",
                    "compare_to_wan",
                    "full_report"
                ], {"default": "distribution"}),
            },
            "optional": {
                "wan_reference": ("LATENT",),  # Optional WAN latent for comparison
            }
        }
    
    RETURN_TYPES = ("STRING", "DICT")
    RETURN_NAMES = ("analysis", "mapping_suggestions")
    FUNCTION = "analyze"
    CATEGORY = "QwenWANBridge/Analysis"
    
    def analyze(self, qwen_latent, analysis_type, wan_reference=None):
        
        analysis = []
        analysis.append(f"Latent Mapping Analysis ({analysis_type})")
        analysis.append("="*50)
        
        # Extract latent
        qwen = qwen_latent["samples"]
        if len(qwen.shape) >= 4:
            qwen = qwen[0]  # Take first batch
        if len(qwen.shape) >= 4:
            qwen = qwen[:, 0]  # Take first frame if temporal
        
        C, H, W = qwen.shape[-3:]
        
        suggestions = {}
        
        if analysis_type in ["distribution", "full_report"]:
            analysis.append("\nDistribution Analysis:")
            analysis.append(f"  Shape: {qwen.shape}")
            analysis.append(f"  Mean: {qwen.mean():.4f}")
            analysis.append(f"  Std: {qwen.std():.4f}")
            analysis.append(f"  Min: {qwen.min():.4f}")
            analysis.append(f"  Max: {qwen.max():.4f}")
            analysis.append(f"  Median: {qwen.median():.4f}")
            
            # Check if normally distributed
            skewness = ((qwen - qwen.mean()) ** 3).mean() / (qwen.std() ** 3)
            analysis.append(f"  Skewness: {skewness:.4f}")
            
            if abs(skewness) < 0.5:
                analysis.append("  → Near normal distribution")
                suggestions["normalization"] = "none_needed"
            else:
                analysis.append("  → Skewed distribution")
                suggestions["normalization"] = "standardize"
        
        if analysis_type in ["channel_analysis", "full_report"]:
            analysis.append("\nChannel-wise Analysis:")
            
            channel_stats = []
            for c in range(C):
                c_mean = qwen[c].mean().item()
                c_std = qwen[c].std().item()
                c_range = qwen[c].max().item() - qwen[c].min().item()
                channel_stats.append((c_mean, c_std, c_range))
                
                if c < 8 or analysis_type == "full_report":
                    analysis.append(f"  Ch{c:2d}: μ={c_mean:+.3f}, σ={c_std:.3f}, range={c_range:.3f}")
            
            # Check channel consistency
            means = [s[0] for s in channel_stats]
            stds = [s[1] for s in channel_stats]
            
            mean_variance = torch.std(torch.tensor(means)).item()
            std_variance = torch.std(torch.tensor(stds)).item()
            
            analysis.append(f"\n  Channel consistency:")
            analysis.append(f"    Mean variance: {mean_variance:.4f}")
            analysis.append(f"    Std variance: {std_variance:.4f}")
            
            if mean_variance > 0.5:
                analysis.append("    → Channels have different distributions")
                suggestions["channel_norm"] = "per_channel"
            else:
                analysis.append("    → Channels are consistent")
                suggestions["channel_norm"] = "global"
        
        if analysis_type in ["spatial_patterns", "full_report"]:
            analysis.append("\nSpatial Pattern Analysis:")
            
            # Check for spatial structure
            center = qwen[:, H//4:3*H//4, W//4:3*W//4]
            edges = torch.cat([
                qwen[:, :H//4, :],
                qwen[:, 3*H//4:, :],
                qwen[:, :, :W//4],
                qwen[:, :, 3*W//4:]
            ], dim=1)
            
            center_energy = center.abs().mean().item()
            edge_energy = edges.abs().mean().item()
            
            analysis.append(f"  Center energy: {center_energy:.4f}")
            analysis.append(f"  Edge energy: {edge_energy:.4f}")
            analysis.append(f"  Ratio: {center_energy/max(edge_energy, 1e-8):.2f}")
            
            if center_energy > edge_energy * 1.5:
                analysis.append("  → Content concentrated in center")
                suggestions["padding"] = "reflection"
            else:
                analysis.append("  → Content distributed evenly")
                suggestions["padding"] = "zeros"
        
        if wan_reference is not None and analysis_type in ["compare_to_wan", "full_report"]:
            analysis.append("\nComparison to WAN Reference:")
            
            wan = wan_reference["samples"]
            if len(wan.shape) >= 4:
                wan = wan[0]
            if len(wan.shape) >= 4:
                wan = wan[:, 0]
            
            # Compare distributions
            qwen_flat = qwen.flatten()
            wan_flat = wan.flatten()
            
            # KL divergence approximation
            qwen_hist = torch.histc(qwen_flat, bins=50)
            wan_hist = torch.histc(wan_flat, bins=50)
            
            qwen_hist = qwen_hist / qwen_hist.sum()
            wan_hist = wan_hist / wan_hist.sum()
            
            kl_div = (qwen_hist * (qwen_hist / (wan_hist + 1e-8)).log()).sum().item()
            
            analysis.append(f"  KL divergence: {kl_div:.4f}")
            
            if kl_div < 0.1:
                analysis.append("  → Distributions are very similar")
                suggestions["mapping"] = "direct"
            elif kl_div < 0.5:
                analysis.append("  → Distributions are somewhat similar")
                suggestions["mapping"] = "linear_transform"
            else:
                analysis.append("  → Distributions are quite different")
                suggestions["mapping"] = "histogram_matching"
            
            # Suggest scaling
            scale = wan.std() / qwen.std()
            shift = wan.mean() - qwen.mean()
            
            analysis.append(f"\n  Suggested transform:")
            analysis.append(f"    Scale: {scale:.3f}")
            analysis.append(f"    Shift: {shift:.3f}")
            suggestions["scale"] = float(scale)
            suggestions["shift"] = float(shift)
        
        # Final recommendations
        analysis.append("\nMapping Recommendations:")
        
        if suggestions.get("mapping") == "direct":
            analysis.append("• Use latents directly, no transformation needed")
        elif suggestions.get("normalization") == "standardize":
            analysis.append("• Standardize to zero mean, unit variance")
        
        if "scale" in suggestions:
            analysis.append(f"• Scale by {suggestions['scale']:.3f}")
            analysis.append(f"• Shift by {suggestions['shift']:.3f}")
        
        analysis.append("\nNoise Application:")
        analysis.append("• Let sampler handle ALL noise via denoise parameter")
        analysis.append("• Do NOT add noise in bridge")
        analysis.append("• For I2V: Use denoise 0.1-0.5")
        analysis.append("• For V2V: Can use higher denoise")
        
        return ("\n".join(analysis), suggestions)