"""
Unified Qwen-WAN Bridge with Text Embedding Alignment
Aligns both latent space AND text embeddings for better compatibility
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class QwenWANUnifiedBridge:
    """
    Complete bridge that aligns:
    1. Qwen latents → WAN latent space
    2. Qwen CLIP embeddings → WAN UMT5 space (via projection)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 4}),
                "alignment_mode": (["minimal", "statistical", "learned"], {"default": "statistical"}),
                "preserve_first_frame": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "qwen_text_embeds": ("CONDITIONING",),  # From Qwen CLIP
                "wan_text_embeds": ("WANVIDTEXT_EMBEDS",),  # Target UMT5 embeddings
                "alignment_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "WANVIDTEXT_EMBEDS", "STRING")
    RETURN_NAMES = ("image_embeds", "aligned_text_embeds", "report")
    FUNCTION = "bridge"
    CATEGORY = "QwenWANBridge"
    
    def bridge(self, qwen_latent, num_frames, alignment_mode, preserve_first_frame,
               qwen_text_embeds=None, wan_text_embeds=None, alignment_strength=0.7):
        
        report = []
        report.append("Unified Qwen-WAN Bridge Report")
        report.append("="*50)
        
        # Extract Qwen latent
        qwen = qwen_latent["samples"]
        if len(qwen.shape) == 5:
            qwen = qwen[0, :, 0, :, :]
        elif len(qwen.shape) == 4:
            qwen = qwen[0]
        
        C, H, W = qwen.shape
        device = qwen.device
        dtype = qwen.dtype
        
        # Align frames to WAN requirements
        original_frames = num_frames
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        temporal_frames = (num_frames - 1) // 4 + 1
        
        report.append(f"\nFrame Configuration:")
        report.append(f"  Requested: {original_frames}")
        report.append(f"  Aligned: {num_frames}")
        report.append(f"  Temporal: {temporal_frames}")
        
        # LATENT ALIGNMENT
        report.append(f"\nLatent Alignment ({alignment_mode}):")
        
        if alignment_mode == "minimal":
            # Direct pass-through
            aligned_latent = qwen
            report.append("  Using Qwen latent directly (no modification)")
            
        elif alignment_mode == "statistical":
            # Match WAN's typical distribution
            # Based on WAN normalization values we discovered
            wan_mean = torch.tensor([
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
            ], device=device, dtype=dtype).view(16, 1, 1)
            
            wan_std = torch.tensor([
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ], device=device, dtype=dtype).view(16, 1, 1)
            
            # Normalize Qwen to standard, then scale to WAN distribution
            qwen_normalized = (qwen - qwen.mean(dim=(1,2), keepdim=True)) / (qwen.std(dim=(1,2), keepdim=True) + 1e-8)
            aligned_latent = qwen_normalized * wan_std + wan_mean
            
            # Blend with original based on alignment_strength
            aligned_latent = alignment_strength * aligned_latent + (1 - alignment_strength) * qwen
            
            report.append(f"  Normalized to WAN distribution")
            report.append(f"  Alignment strength: {alignment_strength}")
            
        elif alignment_mode == "learned":
            # Placeholder for learned projection (would need training)
            # For now, use a simple linear combination that preserves structure
            weights = torch.ones(16, device=device, dtype=dtype) * 0.95
            weights[::2] *= 1.05  # Slightly boost even channels
            aligned_latent = qwen * weights.view(16, 1, 1)
            report.append("  Applied learned channel weights (placeholder)")
        
        # Statistics
        qwen_stats = f"min={qwen.min():.2f}, max={qwen.max():.2f}, mean={qwen.mean():.2f}, std={qwen.std():.2f}"
        aligned_stats = f"min={aligned_latent.min():.2f}, max={aligned_latent.max():.2f}, mean={aligned_latent.mean():.2f}, std={aligned_latent.std():.2f}"
        report.append(f"  Qwen stats: {qwen_stats}")
        report.append(f"  Aligned stats: {aligned_stats}")
        
        # Create temporal extension
        y = torch.zeros(C, temporal_frames, H, W, device=device, dtype=dtype)
        
        if preserve_first_frame:
            # Keep first frame exactly as aligned
            y[:, 0] = aligned_latent
            report.append("\nFirst frame preserved exactly")
        else:
            # Apply some smoothing
            y[:, 0] = aligned_latent * 0.95
            report.append("\nFirst frame slightly smoothed")
        
        # Fill remaining frames (for multi-frame)
        if temporal_frames > 1:
            # Exponential decay - WAN will generate motion from this
            for t in range(1, min(3, temporal_frames)):  # Only first few frames
                decay = 0.9 ** t
                y[:, t] = aligned_latent * decay * 0.1  # Very subtle
            report.append(f"  Added {min(3, temporal_frames)-1} decay frames")
        
        # Create mask
        mask = torch.zeros(1, num_frames, H, W, device=device)
        mask[:, 0] = 1.0  # Only first frame has content
        
        # Reshape mask for WAN
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)
        
        # Ensure correct frame count
        frames_needed = temporal_frames * 4
        if mask.shape[1] < frames_needed:
            padding = torch.zeros(1, frames_needed - mask.shape[1], H, W, device=device)
            mask = torch.cat([mask, padding], dim=1)
        elif mask.shape[1] > frames_needed:
            mask = mask[:, :frames_needed]
        
        mask = mask.view(1, temporal_frames, 4, H, W)
        mask = mask.movedim(1, 2)[0]  # (4, T, H, W)
        
        # TEXT EMBEDDING ALIGNMENT
        aligned_text = None
        if qwen_text_embeds is not None and wan_text_embeds is not None:
            report.append("\nText Embedding Alignment:")
            
            # Extract embeddings
            qwen_cond = qwen_text_embeds[0][0]  # CLIP conditioning
            
            # Simple projection: CLIP → UMT5 space
            # In practice, this would need a learned projection matrix
            # For now, we'll do dimension matching and normalization
            
            if hasattr(wan_text_embeds, 'get'):
                wan_shape = wan_text_embeds.get('shape', None)
                if wan_shape:
                    target_dim = wan_shape[-1]
                    
                    # Project Qwen CLIP to target dimension
                    if qwen_cond.shape[-1] != target_dim:
                        # Simple linear projection (would be learned in practice)
                        projection = F.adaptive_avg_pool1d(
                            qwen_cond.unsqueeze(0).transpose(1, 2),
                            target_dim
                        ).transpose(1, 2).squeeze(0)
                    else:
                        projection = qwen_cond
                    
                    # Normalize to match UMT5 distribution
                    projection = F.normalize(projection, dim=-1) * 2.0
                    
                    aligned_text = {
                        'embeds': projection,
                        'shape': wan_shape,
                    }
                    report.append(f"  Projected CLIP ({qwen_cond.shape}) → UMT5 ({wan_shape})")
            else:
                report.append("  No text alignment (missing target shape)")
        elif qwen_text_embeds is not None:
            report.append("\nText embeddings provided but no target for alignment")
        
        # Create image embeds structure
        image_embeds = {
            "image_embeds": y,
            "mask": mask,
            "num_frames": num_frames,
            "lat_h": H,
            "lat_w": W,
            "target_shape": (C, temporal_frames, H, W),
            "has_ref": False,
            "fun_or_fl2v_model": False,
            "clip_context": None,
            "negative_clip_context": None,
            "control_embeds": None,
            "add_cond_latents": None,
            "end_image": None,
            "max_seq_len": H * W // 4 * temporal_frames,
        }
        
        report_text = "\n".join(report)
        
        return (image_embeds, aligned_text or {}, report_text)


class QwenWANProgressiveTester:
    """
    Test bridge with progressive frame counts to find the sweet spot
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bridge_node": ("QWEN_WAN_BRIDGE",),  # Any of our bridge nodes
                "qwen_latent": ("LATENT",),
                "test_frames": ("STRING", {"default": "1,5,9,13,17", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"
    CATEGORY = "QwenWANBridge"
    
    def test(self, bridge_node, qwen_latent, test_frames):
        results = []
        results.append("Progressive Frame Test Results")
        results.append("="*50)
        
        frame_counts = [int(x.strip()) for x in test_frames.split(",")]
        
        for num_frames in frame_counts:
            results.append(f"\nTesting {num_frames} frames:")
            
            try:
                # Call bridge with different frame counts
                output = bridge_node.bridge(
                    qwen_latent=qwen_latent,
                    num_frames=num_frames,
                    alignment_mode="statistical",
                    preserve_first_frame=True
                )
                
                if output and len(output) > 0:
                    image_embeds = output[0]
                    shape = image_embeds.get("image_embeds").shape
                    results.append(f"  Success: Output shape {shape}")
                    
                    # Check first frame preservation
                    first_frame = image_embeds["image_embeds"][:, 0]
                    frame_mean = first_frame.mean().item()
                    frame_std = first_frame.std().item()
                    results.append(f"  First frame: mean={frame_mean:.3f}, std={frame_std:.3f}")
                else:
                    results.append(f"  Failed: No output")
                    
            except Exception as e:
                results.append(f"  Error: {str(e)}")
        
        results.append("\nRecommendation:")
        results.append("Start with num_frames=1 to verify basic compatibility")
        results.append("Then try 5, 9, 13 for short sequences")
        results.append("If those work, scale up to 81+ for full videos")
        
        return ("\n".join(results),)