"""
Bridge specifically for WAN I2V conditioning
Handles the concat_latent_image format that WAN I2V expects
"""

import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils
import node_helpers

class QwenWANI2VBridge:
    """
    Creates proper I2V conditioning from Qwen latents
    Outputs conditioning that can be used directly with KSampler
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1024, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "info")
    FUNCTION = "create_i2v_conditioning"
    CATEGORY = "QwenWANBridge/I2V"
    
    def create_i2v_conditioning(self, qwen_latent, positive, negative, width, height, 
                                num_frames, batch_size, clip_vision_output=None):
        
        info = []
        info.append("I2V Conditioning Bridge")
        info.append("="*40)
        
        # Extract Qwen latent
        qwen = qwen_latent["samples"]
        info.append(f"Input Qwen shape: {qwen.shape}")
        
        # Handle batch dimension - Qwen should be (B, C, H, W) or (B, C, 1, H, W)
        if len(qwen.shape) == 5:
            # (B, C, F, H, W) - take first frame
            qwen = qwen[:, :, 0, :, :]
        elif len(qwen.shape) == 4:
            # (B, C, H, W) - already correct
            pass
        else:
            raise ValueError(f"Unexpected Qwen latent shape: {qwen.shape}")
        
        B, C, H_latent, W_latent = qwen.shape
        info.append(f"Processing shape: {qwen.shape}")
        
        # Create empty latent for video generation
        # WAN expects: (B, 16, T, H, W) where T = ((num_frames - 1) // 4) + 1
        T = ((num_frames - 1) // 4) + 1
        latent = torch.zeros(
            [batch_size, 16, T, height // 8, width // 8], 
            device=comfy.model_management.intermediate_device()
        )
        info.append(f"Output latent shape: {latent.shape}")
        
        # Resize Qwen latent if needed
        target_H = height // 8
        target_W = width // 8
        
        if H_latent != target_H or W_latent != target_W:
            # Resize using interpolation
            qwen_resized = F.interpolate(
                qwen,
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=False
            )
            info.append(f"Resized Qwen to: {qwen_resized.shape}")
        else:
            qwen_resized = qwen
        
        # Handle channel mismatch (Qwen has 16 channels, WAN expects 16 - perfect!)
        if C != 16:
            info.append(f"WARNING: Qwen has {C} channels, expected 16")
            if C < 16:
                # Pad with zeros
                padding = torch.zeros(B, 16 - C, target_H, target_W, device=qwen.device)
                qwen_resized = torch.cat([qwen_resized, padding], dim=1)
            else:
                # Truncate
                qwen_resized = qwen_resized[:, :16]
        
        # Create concat_latent_image for I2V conditioning
        # This needs to be in video format: (B, C, T, H, W)
        concat_latent_image = torch.zeros(
            [batch_size, 16, T, target_H, target_W],
            device=qwen_resized.device,
            dtype=qwen_resized.dtype
        )
        
        # Set the first frame(s) to our Qwen latent
        # We can repeat the Qwen frame for the first few timesteps
        start_frames = min(3, T)  # Use first 3 timesteps or less
        for t in range(start_frames):
            concat_latent_image[:, :, t] = qwen_resized[0] if B == 1 else qwen_resized[min(t, B-1)]
        
        info.append(f"Set {start_frames} frames with Qwen latent")
        
        # Create mask - 0 where we have content, 1 where we need generation
        mask = torch.ones(
            (1, 1, T, target_H, target_W),
            device=qwen_resized.device,
            dtype=qwen_resized.dtype
        )
        mask[:, :, :start_frames] = 0.0  # Don't modify the frames we set
        info.append(f"Mask shape: {mask.shape}, masked {start_frames} frames")
        
        # Apply conditioning
        positive_out = node_helpers.conditioning_set_values(
            positive, 
            {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative_out = node_helpers.conditioning_set_values(
            negative,
            {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        
        # Add clip vision if provided
        if clip_vision_output is not None:
            positive_out = node_helpers.conditioning_set_values(
                positive_out,
                {"clip_vision_output": clip_vision_output}
            )
            negative_out = node_helpers.conditioning_set_values(
                negative_out, 
                {"clip_vision_output": clip_vision_output}
            )
            info.append("Added CLIP vision conditioning")
        
        # Prepare output latent
        out_latent = {"samples": latent}
        
        info.append("I2V conditioning created successfully")
        
        return (positive_out, negative_out, out_latent, "\n".join(info))


class QwenToImage:
    """
    Decodes Qwen latent to image for use with standard WanImageToVideo
    Simple VAE decode approach
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "vae": ("VAE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "QwenWANBridge/Utils"
    
    def decode(self, qwen_latent, vae):
        # Extract samples
        samples = qwen_latent["samples"]
        
        # Handle different shapes
        if len(samples.shape) == 5:
            # (B, C, F, H, W) - take first frame
            samples = samples[:, :, 0, :, :]
        elif len(samples.shape) == 4:
            # (B, C, H, W) - already correct
            pass
        else:
            # (C, H, W) - add batch dimension
            samples = samples.unsqueeze(0)
        
        # Decode using VAE
        image = vae.decode(samples)
        
        return (image,)


class QwenWANI2VDirect:
    """
    Direct latent-to-latent I2V setup without VAE decode/encode
    Preserves maximum quality by staying in latent space
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "model": ("MODEL",),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1024, "step": 4}),
                "mode": (["direct", "reference", "hybrid"], {"default": "direct"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "prepare"
    CATEGORY = "QwenWANBridge/I2V"
    
    def prepare(self, qwen_latent, model, width, height, num_frames, mode, strength):
        info = []
        info.append(f"Direct I2V Bridge - Mode: {mode}")
        info.append("="*40)
        
        # Extract Qwen latent
        qwen = qwen_latent["samples"]
        
        # Handle shapes
        if len(qwen.shape) == 5:
            qwen = qwen[:, :, 0, :, :]
        elif len(qwen.shape) != 4:
            qwen = qwen.unsqueeze(0)
        
        B, C, H_latent, W_latent = qwen.shape
        info.append(f"Qwen shape: {qwen.shape}")
        
        # Create video latent
        T = ((num_frames - 1) // 4) + 1
        target_H = height // 8
        target_W = width // 8
        
        # Resize if needed
        if H_latent != target_H or W_latent != target_W:
            qwen = F.interpolate(qwen, size=(target_H, target_W), mode='bilinear', align_corners=False)
            info.append(f"Resized to: {qwen.shape}")
        
        # Create output based on mode
        if mode == "direct":
            # Direct copy to all frames
            latent = qwen.unsqueeze(2).repeat(1, 1, T, 1, 1)
            latent = latent * strength  # Apply strength
            info.append(f"Direct mode: copied to all {T} frames")
            
        elif mode == "reference":
            # Only first frame, rest is noise
            latent = torch.randn(B, 16, T, target_H, target_W, device=qwen.device) * 0.1
            latent[:, :, 0] = qwen[:, :16] if C >= 16 else F.pad(qwen, (0, 0, 0, 0, 0, 16-C))
            info.append("Reference mode: first frame only")
            
        elif mode == "hybrid":
            # Decay over time
            latent = torch.zeros(B, 16, T, target_H, target_W, device=qwen.device)
            qwen_16 = qwen[:, :16] if C >= 16 else F.pad(qwen, (0, 0, 0, 0, 0, 16-C))
            for t in range(T):
                decay = strength * (1.0 - t / T)
                noise = torch.randn_like(qwen_16) * (1.0 - decay)
                latent[:, :, t] = qwen_16 * decay + noise
            info.append("Hybrid mode: decaying influence")
        
        out = {"samples": latent}
        info.append(f"Output shape: {latent.shape}")
        
        return (out, "\n".join(info))