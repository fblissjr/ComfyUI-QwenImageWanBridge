"""
Qwen to WAN T2I-style Bridge
Treats Qwen output as if it were WAN's own T2I generation
"""

import torch
import torch.nn.functional as F

class QwenWANT2IBridge:
    """
    Bridge that treats Qwen latents as if they were WAN T2I outputs
    Since Qwen-Image produces outputs aligned with WAN T2I, we can
    use them directly for I2V extension
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1024, "step": 4}),
                "latent_mode": (["direct", "normalized", "scaled"], {"default": "direct"}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "temporal_fill": (["zeros", "noise", "repeat_decay"], {"default": "zeros"}),
            },
            "optional": {
                "wan_t2i_reference": ("LATENT",),  # Optional WAN T2I output for comparison
            }
        }
    
    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "STRING")
    RETURN_NAMES = ("image_embeds", "analysis")
    FUNCTION = "bridge"
    CATEGORY = "QwenWANBridge"
    
    def bridge(self, qwen_latent, num_frames, latent_mode, scale_factor, 
               temporal_fill, wan_t2i_reference=None):
        
        # Extract Qwen latent
        qwen = qwen_latent["samples"]
        
        # Handle shapes
        if len(qwen.shape) == 5:
            qwen = qwen[0, :, 0, :, :]  # Take first frame
        elif len(qwen.shape) == 4:
            qwen = qwen[0]
        
        C, H, W = qwen.shape
        device = qwen.device
        dtype = qwen.dtype
        
        # Align frames
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        temporal_frames = (num_frames - 1) // 4 + 1
        
        analysis = []
        analysis.append(f"Qwen T2I-style Bridge Analysis")
        analysis.append(f"{'='*40}")
        analysis.append(f"Input shape: {qwen.shape}")
        analysis.append(f"Frames: {num_frames} → {temporal_frames} temporal")
        
        # Analyze Qwen latent statistics
        qwen_stats = {
            "min": qwen.min().item(),
            "max": qwen.max().item(),
            "mean": qwen.mean().item(),
            "std": qwen.std().item(),
        }
        analysis.append(f"\nQwen latent stats:")
        analysis.append(f"  Range: [{qwen_stats['min']:.3f}, {qwen_stats['max']:.3f}]")
        analysis.append(f"  Mean: {qwen_stats['mean']:.3f}, Std: {qwen_stats['std']:.3f}")
        
        # If WAN reference provided, compare
        if wan_t2i_reference is not None:
            wan_ref = wan_t2i_reference["samples"]
            if len(wan_ref.shape) == 5:
                wan_ref = wan_ref[0, :, 0, :, :]
            elif len(wan_ref.shape) == 4:
                wan_ref = wan_ref[0]
            
            wan_stats = {
                "min": wan_ref.min().item(),
                "max": wan_ref.max().item(),
                "mean": wan_ref.mean().item(),
                "std": wan_ref.std().item(),
            }
            analysis.append(f"\nWAN T2I reference stats:")
            analysis.append(f"  Range: [{wan_stats['min']:.3f}, {wan_stats['max']:.3f}]")
            analysis.append(f"  Mean: {wan_stats['mean']:.3f}, Std: {wan_stats['std']:.3f}")
            
            # Calculate alignment factor
            scale_suggestion = wan_stats['std'] / qwen_stats['std'] if qwen_stats['std'] > 0 else 1.0
            shift_suggestion = wan_stats['mean'] - qwen_stats['mean']
            analysis.append(f"\nAlignment suggestions:")
            analysis.append(f"  Scale factor: {scale_suggestion:.3f}")
            analysis.append(f"  Shift: {shift_suggestion:.3f}")
        
        # Process latent based on mode
        if latent_mode == "direct":
            # Use Qwen latent as-is (since it's aligned with WAN T2I)
            processed = qwen
            
        elif latent_mode == "normalized":
            # Normalize to standard distribution then scale to WAN range
            # WAN T2I typically has std around 1.5-2.5
            target_std = 2.0
            processed = (qwen - qwen.mean()) / (qwen.std() + 1e-8) * target_std
            
        elif latent_mode == "scaled":
            # Simple scaling
            processed = qwen * scale_factor
        
        analysis.append(f"\nProcessing mode: {latent_mode}")
        if latent_mode != "direct":
            proc_stats = {
                "min": processed.min().item(),
                "max": processed.max().item(),
                "mean": processed.mean().item(),
                "std": processed.std().item(),
            }
            analysis.append(f"Processed stats:")
            analysis.append(f"  Range: [{proc_stats['min']:.3f}, {proc_stats['max']:.3f}]")
            analysis.append(f"  Mean: {proc_stats['mean']:.3f}, Std: {proc_stats['std']:.3f}")
        
        # Create temporal extension
        y = torch.zeros(C, temporal_frames, H, W, device=device, dtype=dtype)
        
        # Place processed frame as first frame
        y[:, 0] = processed
        
        # Fill temporal dimension
        if temporal_frames > 1:
            if temporal_fill == "zeros":
                # Already zeros
                pass
            elif temporal_fill == "noise":
                # Fill with noise matching processed statistics
                for t in range(1, temporal_frames):
                    noise = torch.randn_like(processed) * processed.std()
                    y[:, t] = noise + processed.mean()
            elif temporal_fill == "repeat_decay":
                # Repeat with exponential decay
                for t in range(1, temporal_frames):
                    decay = 0.9 ** t
                    y[:, t] = processed * decay
        
        # Create mask - only first frame has content for I2V
        mask = torch.zeros(1, num_frames, H, W, device=device)
        mask[:, 0] = 1.0
        
        # Reshape mask for WAN
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)
        
        # Pad or trim mask to match temporal frames
        frames_needed = temporal_frames * 4
        if mask.shape[1] < frames_needed:
            padding = torch.zeros(1, frames_needed - mask.shape[1], H, W, device=device)
            mask = torch.cat([mask, padding], dim=1)
        elif mask.shape[1] > frames_needed:
            mask = mask[:, :frames_needed]
        
        # Reshape into groups of 4
        mask = mask.view(1, temporal_frames, 4, H, W)
        mask = mask.movedim(1, 2)[0]  # (4, T, H, W)
        
        analysis.append(f"\nOutput shapes:")
        analysis.append(f"  Latent: {y.shape}")
        analysis.append(f"  Mask: {mask.shape}")
        
        # Create I2V embedding structure
        image_embeds = {
            "image_embeds": y,
            "mask": mask,
            "num_frames": num_frames,
            "lat_h": H,
            "lat_w": W,
            "target_shape": (C, temporal_frames, H, W),
            "has_ref": False,
            "fun_or_fl2v_model": False,
            # Minimal metadata for I2V
            "clip_context": None,
            "negative_clip_context": None,
            "control_embeds": None,
            "add_cond_latents": None,
            "end_image": None,
            "max_seq_len": H * W // 4 * temporal_frames,
        }
        
        analysis_text = "\n".join(analysis)
        
        return (image_embeds, analysis_text)


class QwenWANLatentComparator:
    """
    Compare Qwen and WAN latents to understand alignment
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "wan_latent": ("LATENT",),
                "channel_to_visualize": ("INT", {"default": 0, "min": 0, "max": 15}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("comparison_image", "statistics")
    FUNCTION = "compare"
    CATEGORY = "QwenWANBridge"
    
    def compare(self, qwen_latent, wan_latent, channel_to_visualize):
        import numpy as np
        
        # Extract latents
        qwen = qwen_latent["samples"]
        wan = wan_latent["samples"]
        
        # Get first frame from each
        if len(qwen.shape) == 5:
            qwen = qwen[0, :, 0, :, :]
        elif len(qwen.shape) == 4:
            qwen = qwen[0]
            
        if len(wan.shape) == 5:
            wan = wan[0, :, 0, :, :]
        elif len(wan.shape) == 4:
            wan = wan[0]
        
        # Ensure same spatial dimensions
        if qwen.shape[-2:] != wan.shape[-2:]:
            wan = F.interpolate(wan.unsqueeze(0), size=qwen.shape[-2:], 
                               mode='bilinear', align_corners=False).squeeze(0)
        
        # Calculate statistics
        stats = []
        stats.append("Latent Comparison Statistics")
        stats.append("="*40)
        
        # Overall statistics
        stats.append("\nOverall Statistics:")
        stats.append(f"Qwen shape: {qwen.shape}")
        stats.append(f"WAN shape:  {wan.shape}")
        stats.append(f"\nQwen - Range: [{qwen.min():.3f}, {qwen.max():.3f}]")
        stats.append(f"       Mean: {qwen.mean():.3f}, Std: {qwen.std():.3f}")
        stats.append(f"\nWAN -  Range: [{wan.min():.3f}, {wan.max():.3f}]")
        stats.append(f"       Mean: {wan.mean():.3f}, Std: {wan.std():.3f}")
        
        # Channel-wise comparison
        stats.append("\nChannel-wise Correlation:")
        correlations = []
        for c in range(min(qwen.shape[0], wan.shape[0])):
            q_chan = qwen[c].flatten()
            w_chan = wan[c].flatten()
            if q_chan.std() > 0 and w_chan.std() > 0:
                corr = torch.corrcoef(torch.stack([q_chan, w_chan]))[0, 1].item()
                correlations.append(corr)
                if c < 8:  # Show first 8 channels
                    stats.append(f"  Channel {c:2d}: {corr:+.3f}")
        
        avg_corr = sum(correlations) / len(correlations) if correlations else 0
        stats.append(f"\nAverage correlation: {avg_corr:.3f}")
        
        # Distribution alignment
        qwen_norm = (qwen - qwen.mean()) / (qwen.std() + 1e-8)
        wan_norm = (wan - wan.mean()) / (wan.std() + 1e-8)
        norm_diff = (qwen_norm - wan_norm).abs().mean().item()
        stats.append(f"\nNormalized difference: {norm_diff:.3f}")
        stats.append("(Lower is better, <0.5 is good alignment)")
        
        # Create visualization
        c = min(channel_to_visualize, qwen.shape[0]-1, wan.shape[0]-1)
        
        # Normalize for visualization
        def normalize_for_vis(x):
            x_min = x.min()
            x_max = x.max()
            if x_max > x_min:
                return (x - x_min) / (x_max - x_min)
            return torch.zeros_like(x)
        
        qwen_vis = normalize_for_vis(qwen[c])
        wan_vis = normalize_for_vis(wan[c])
        diff_vis = (qwen_vis - wan_vis).abs()
        
        # Stack horizontally: Qwen | WAN | Difference
        vis = torch.cat([qwen_vis, wan_vis, diff_vis], dim=1)
        
        # Convert to RGB image
        vis = vis.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        vis = vis.repeat(1, 3, 1, 1)  # Convert to RGB
        vis = vis.cpu().numpy().transpose(0, 2, 3, 1)
        
        stats_text = "\n".join(stats)
        
        return (vis, stats_text)