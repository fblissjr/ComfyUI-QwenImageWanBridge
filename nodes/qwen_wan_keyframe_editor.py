"""
Qwen-WAN Keyframe Editor Node
Combines Qwen's vision-aware editing with WAN's temporal generation
Built on lessons from research nodes
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

# ComfyUI imports
try:
    from comfy.utils import common_upscale
    from comfy import model_management as mm
    device = mm.intermediate_device()
    offload_device = mm.unet_offload_device()
except:
    # Fallback for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offload_device = torch.device("cpu")
    def common_upscale(x, w, h, method, crop):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)


class QwenWANKeyframeEditor:
    """
    Edit video keyframes using Qwen Image Edit's vision understanding
    Then regenerate video with WAN's temporal coherence
    
    Combines best of both:
    - Qwen2.5-VL vision tokens for understanding edits
    - WAN2.1's 16-channel compatibility for direct integration
    - Semantic bridging between UMT5 and Qwen2.5-VL
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Video input (from WAN generation or external)
                "video_frames": ("IMAGE",),  # Batch of frames
                "keyframe_indices": ("STRING", {
                    "default": "0,20,40,60,80",
                    "multiline": False
                }),
                "edit_prompts": ("STRING", {
                    "default": "make the car red|add sunglasses|change background to sunset|add motion blur|enhance colors",
                    "multiline": True
                }),
                
                # Bridge settings
                "bridge_mode": (["semantic", "latent_direct", "whisk_style"], {"default": "semantic"}),
                "alignment_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                
                # V2V settings
                "denoise_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temporal_blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                
                # Frame settings
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
            },
            "optional": {
                # Optional Qwen conditioning for consistent editing
                "qwen_conditioning": ("CONDITIONING",),
                # Optional WAN embeddings for style preservation  
                "wan_embeddings": ("WANVIDTEXT_EMBEDS",),
                # Vision context from Qwen2.5-VL
                "vision_context": ("VLM_CONTEXT",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "IMAGE", "WANVIDIMAGE_EMBEDS", "STRING")
    RETURN_NAMES = ("edited_latents", "preview_frames", "video_embeds", "edit_report")
    FUNCTION = "edit_keyframes"
    CATEGORY = "QwenWANBridge/KeyframeEdit"
    
    def edit_keyframes(self, video_frames, keyframe_indices, edit_prompts, 
                      bridge_mode, alignment_strength, denoise_strength, 
                      temporal_blend, width, height,
                      qwen_conditioning=None, wan_embeddings=None, vision_context=None):
        
        report = []
        report.append("=== Qwen-WAN Keyframe Editor Report ===")
        
        # Parse inputs
        indices = [int(i.strip()) for i in keyframe_indices.split(',')]
        prompts = [p.strip() for p in edit_prompts.split('|')]
        
        # Ensure we have enough prompts
        while len(prompts) < len(indices):
            prompts.append(prompts[-1] if prompts else "enhance quality")
        
        report.append(f"\nKeyframe Configuration:")
        report.append(f"  Frames to edit: {indices}")
        report.append(f"  Edit prompts: {len(prompts)} provided")
        report.append(f"  Bridge mode: {bridge_mode}")
        
        # Extract keyframes
        B, H_orig, W_orig, C = video_frames.shape
        keyframes = []
        for idx in indices:
            if idx < B:
                keyframes.append(video_frames[idx:idx+1])
            else:
                report.append(f"  Warning: Frame {idx} out of range (max {B-1})")
        
        if not keyframes:
            report.append("ERROR: No valid keyframes to edit!")
            return (None, None, None, '\n'.join(report))
        
        report.append(f"  Extracted {len(keyframes)} keyframes")
        
        # Process based on bridge mode
        if bridge_mode == "semantic":
            edited_frames, embeds = self.semantic_bridge_edit(
                keyframes, prompts, vision_context, report
            )
        elif bridge_mode == "latent_direct":
            edited_frames, embeds = self.latent_bridge_edit(
                keyframes, prompts, alignment_strength, report
            )
        else:  # whisk_style
            edited_frames, embeds = self.whisk_bridge_edit(
                keyframes, prompts, report
            )
        
        # Create temporal masks for blending
        temporal_masks = self.create_temporal_masks(
            B, indices, denoise_strength, temporal_blend, report
        )
        
        # Prepare output latents (combining edited keyframes with original)
        output_latents = self.blend_keyframes(
            video_frames, edited_frames, indices, temporal_masks, report
        )
        
        # Create preview frames (for visualization)
        preview_frames = self.create_preview(edited_frames, report)
        
        # Generate WAN embeddings for V2V
        lat_h, lat_w = height // 8, width // 8
        video_embeds = self.create_wan_embeddings(
            output_latents, lat_h, lat_w, len(indices), report
        )
        
        report.append("\n=== Processing Complete ===")
        return (output_latents, preview_frames, video_embeds, '\n'.join(report))
    
    def semantic_bridge_edit(self, keyframes, prompts, vision_context, report):
        """
        Use semantic understanding to bridge Qwen edits to WAN
        Most reliable approach - uses text as intermediate
        """
        report.append("\nSemantic Bridge Mode:")
        
        edited_frames = []
        embeddings = []
        
        for frame, prompt in zip(keyframes, prompts):
            # Here we would:
            # 1. Use Qwen2.5-VL to understand the frame
            # 2. Apply the edit semantically
            # 3. Generate description for WAN
            
            report.append(f"  Edit: '{prompt[:30]}...'")
            
            # Placeholder: Apply basic transformation
            edited = self.apply_placeholder_edit(frame, prompt)
            edited_frames.append(edited)
            
            # Generate semantic embedding (placeholder)
            embed = self.generate_semantic_embedding(edited, prompt)
            embeddings.append(embed)
        
        report.append(f"  Processed {len(edited_frames)} semantic edits")
        return edited_frames, embeddings
    
    def latent_bridge_edit(self, keyframes, prompts, alignment_strength, report):
        """
        Direct latent space bridging with alignment
        Based on qwen_wan_unified_bridge discoveries
        """
        report.append("\nLatent Bridge Mode:")
        
        # WAN normalization values (from research)
        wan_mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ], device=device).view(16, 1, 1)
        
        wan_std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ], device=device).view(16, 1, 1)
        
        edited_frames = []
        embeddings = []
        
        for frame, prompt in zip(keyframes, prompts):
            # Apply edit (placeholder)
            edited = self.apply_placeholder_edit(frame, prompt)
            
            # Convert to latent and align
            if len(edited.shape) == 4:  # B,H,W,C
                edited_latent = edited.permute(0, 3, 1, 2)  # B,C,H,W
            else:
                edited_latent = edited
            
            # Normalize to WAN distribution
            normalized = (edited_latent - edited_latent.mean()) / (edited_latent.std() + 1e-8)
            aligned = normalized * wan_std[:3] + wan_mean[:3]  # RGB channels only
            
            # Blend based on alignment strength
            final = alignment_strength * aligned + (1 - alignment_strength) * edited_latent
            
            edited_frames.append(final)
            report.append(f"  Aligned edit: '{prompt[:30]}...'")
        
        report.append(f"  Applied latent alignment (strength={alignment_strength})")
        return edited_frames, None
    
    def whisk_bridge_edit(self, keyframes, prompts, report):
        """
        Google Whisk-style: Generate descriptions for editing
        Most compatible but less precise
        """
        report.append("\nWhisk-Style Bridge Mode:")
        
        descriptions = []
        edited_frames = []
        
        for frame, prompt in zip(keyframes, prompts):
            # Generate description of desired edit
            desc = self.generate_edit_description(frame, prompt)
            descriptions.append(desc)
            
            # Apply placeholder edit
            edited = self.apply_placeholder_edit(frame, prompt)
            edited_frames.append(edited)
            
            report.append(f"  Generated: '{desc[:50]}...'")
        
        report.append(f"  Created {len(descriptions)} edit descriptions")
        return edited_frames, descriptions
    
    def create_temporal_masks(self, total_frames, keyframe_indices, denoise, blend, report):
        """Create masks for temporal blending between keyframes"""
        masks = torch.zeros(total_frames, device=device)
        
        for idx in keyframe_indices:
            if idx < total_frames:
                masks[idx] = 1.0
                
                # Add temporal falloff
                for offset in range(1, int(blend * 10) + 1):
                    weight = 1.0 - (offset / (blend * 10))
                    weight *= denoise
                    
                    if idx + offset < total_frames:
                        masks[idx + offset] = max(masks[idx + offset], weight * 0.5)
                    if idx - offset >= 0:
                        masks[idx - offset] = max(masks[idx - offset], weight * 0.5)
        
        report.append(f"\nTemporal Masks:")
        report.append(f"  Denoise: {denoise}, Blend: {blend}")
        report.append(f"  Active frames: {(masks > 0).sum().item()}")
        
        return masks
    
    def blend_keyframes(self, original_frames, edited_frames, indices, masks, report):
        """Blend edited keyframes back into video"""
        B, H, W, C = original_frames.shape
        output = original_frames.clone()
        
        for edited, idx in zip(edited_frames, indices):
            if idx < B:
                # Convert edited frame to correct format
                if len(edited.shape) == 4 and edited.shape[0] == 1:
                    edited = edited[0]
                if len(edited.shape) == 3 and edited.shape[0] == 3:
                    edited = edited.permute(1, 2, 0)
                
                # Ensure correct shape
                if edited.shape != (H, W, C):
                    edited = F.interpolate(
                        edited.unsqueeze(0).permute(0, 3, 1, 2),
                        size=(H, W),
                        mode='bilinear'
                    ).permute(0, 2, 3, 1)[0]
                
                # Blend based on mask
                if idx < len(masks):
                    weight = masks[idx].item()
                    output[idx] = edited * weight + output[idx] * (1 - weight)
                else:
                    output[idx] = edited
        
        report.append(f"  Blended {len(edited_frames)} keyframes into video")
        
        # Convert to latent format
        latent_output = {
            "samples": output.permute(0, 3, 1, 2).unsqueeze(0)  # B,C,H,W -> 1,B,C,H,W
        }
        
        return latent_output
    
    def create_wan_embeddings(self, latents, lat_h, lat_w, num_keyframes, report):
        """Create WAN-compatible embeddings from edited latents"""
        
        # Extract latent tensor
        if isinstance(latents, dict):
            tensor = latents["samples"]
        else:
            tensor = latents
        
        # Get dimensions
        if len(tensor.shape) == 5:
            B, C, F, H, W = tensor.shape
            temporal_frames = F
        else:
            B, C, H, W = tensor.shape
            temporal_frames = 1
        
        # Create mask showing which frames are edited
        mask = torch.zeros(1, temporal_frames, lat_h, lat_w, device=device)
        # Mark keyframes (simplified - would be more complex in practice)
        mask[:, :num_keyframes] = 1.0
        
        # Reshape mask for WAN (following discoveries from bridge_v2)
        mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        mask_final = torch.cat([mask_repeated, mask[:, 1:]], dim=1)
        
        # Calculate max sequence length
        patches_per_frame = lat_h * lat_w // 4  # PATCH_SIZE = (1, 2, 2)
        temporal_patches = (temporal_frames - 1) // 4 + 1
        max_seq_len = temporal_patches * patches_per_frame
        
        # Build embeddings dictionary
        embeddings = {
            "image_embeds": tensor[0] if len(tensor.shape) == 5 else tensor,
            "mask": mask_final,
            "max_seq_len": max_seq_len,
            "num_frames": temporal_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "has_ref": True,  # We have edited keyframes as references
        }
        
        report.append(f"\nWAN Embeddings Created:")
        report.append(f"  Shape: {embeddings['image_embeds'].shape}")
        report.append(f"  Max seq length: {max_seq_len}")
        
        return embeddings
    
    def create_preview(self, edited_frames, report):
        """Create preview grid of edited keyframes"""
        if not edited_frames:
            return None
        
        # Stack frames for preview
        preview = []
        for frame in edited_frames:
            if len(frame.shape) == 4:
                preview.append(frame[0])
            elif len(frame.shape) == 3:
                preview.append(frame)
            else:
                continue
        
        if preview:
            if len(preview[0].shape) == 3 and preview[0].shape[0] in [3, 4]:
                # C,H,W -> H,W,C
                preview = [f.permute(1, 2, 0) for f in preview]
            
            # Stack vertically
            preview = torch.cat(preview, dim=0) if len(preview) > 1 else preview[0]
            report.append(f"  Created preview: {preview.shape}")
        else:
            preview = None
            report.append("  No preview available")
        
        return preview
    
    # Placeholder methods (would be implemented with actual models)
    def apply_placeholder_edit(self, frame, prompt):
        """Placeholder for actual Qwen Image Edit"""
        # In practice, this would call Qwen Image Edit
        # For now, just return frame with slight modification
        edited = frame.clone()
        if "red" in prompt.lower():
            if len(edited.shape) == 4:  # B,H,W,C
                edited[..., 0] *= 1.2  # Boost red channel
            elif len(edited.shape) == 3:  # H,W,C
                edited[..., 0] *= 1.2
        return edited
    
    def generate_semantic_embedding(self, frame, prompt):
        """Placeholder for semantic embedding generation"""
        # Would use Qwen2.5-VL to generate semantic description
        return f"Frame edited with: {prompt}"
    
    def generate_edit_description(self, frame, prompt):
        """Generate text description of edit for Whisk-style approach"""
        # Would use VLM to describe the edited result
        return f"Image showing {prompt}"


class QwenWANKeyframeExtractor:
    """
    Helper node to extract specific frames from video for editing
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "frame_indices": ("STRING", {"default": "0,20,40,60,80"}),
                "extract_mode": (["indices", "uniform", "motion_based"], {"default": "indices"}),
                "num_keyframes": ("INT", {"default": 5, "min": 1, "max": 50}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("keyframes", "indices_used")
    FUNCTION = "extract"
    CATEGORY = "QwenWANBridge/KeyframeEdit"
    
    def extract(self, video_frames, frame_indices, extract_mode, num_keyframes):
        B, H, W, C = video_frames.shape
        
        if extract_mode == "indices":
            indices = [int(i.strip()) for i in frame_indices.split(',')]
        elif extract_mode == "uniform":
            step = max(1, B // num_keyframes)
            indices = list(range(0, B, step))[:num_keyframes]
        else:  # motion_based
            # Placeholder - would analyze motion
            indices = [0, B//4, B//2, 3*B//4, B-1][:num_keyframes]
        
        # Extract frames
        keyframes = []
        valid_indices = []
        for idx in indices:
            if 0 <= idx < B:
                keyframes.append(video_frames[idx:idx+1])
                valid_indices.append(idx)
        
        if keyframes:
            output = torch.cat(keyframes, dim=0)
        else:
            output = video_frames[:1]  # Fallback to first frame
            valid_indices = [0]
        
        return (output, ','.join(map(str, valid_indices)))


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenWANKeyframeEditor": QwenWANKeyframeEditor,
    "QwenWANKeyframeExtractor": QwenWANKeyframeExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenWANKeyframeEditor": "Qwen-WAN Keyframe Editor",
    "QwenWANKeyframeExtractor": "Extract Keyframes for Editing",
}