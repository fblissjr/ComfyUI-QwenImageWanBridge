"""
Unified I2V Bridge - The Swiss Army Knife
Combines all functionality into one intuitive node
"""

import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils
import node_helpers

class QwenWANUnifiedI2V:
    """
    The complete I2V solution - all options in one place
    Works with both standard latents and our custom bridge outputs
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1024, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                
                # Core mode selection
                "i2v_mode": ([
                    "standard",           # Normal I2V conditioning
                    "direct_latent",      # Direct latent injection
                    "reference",          # Reference/phantom mode
                    "hybrid",             # Mix of both
                    "vace_style",         # VACE-like conditioning
                ], {"default": "standard"}),
                
                # Noise control
                "noise_mode": ([
                    "no_noise",           # Pure input
                    "add_noise",          # Add noise on top
                    "mix_noise",          # Blend with noise
                    "scaled_noise",       # Scale by frame
                    "decay_noise",        # Decay over time
                ], {"default": "no_noise"}),
                
                "noise_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                
                # Frame control
                "start_frames": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "frame_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # WAN version handling
                "wan_version": (["auto", "wan21", "wan22"], {"default": "auto"}),
                
                # Channel expansion method for WAN 2.2
                "channel_mode": ([
                    "frequency",  # Frequency-based expansion
                    "repeat",     # Simple 3x repeat
                    "zero_pad",   # Pad with zeros
                ], {"default": "frequency"}),
                
                # Normalization
                "apply_wan_norm": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Multiple input options
                "qwen_latent": ("LATENT",),          # From Qwen encoder
                "bridge_latent": ("LATENT",),        # From our bridge nodes
                "start_image": ("IMAGE",),           # Direct image input
                "vae": ("VAE",),                     # For image encoding if needed
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "info")
    FUNCTION = "process"
    CATEGORY = "QwenWANBridge/Unified"
    
    def process(self, positive, negative, width, height, num_frames, batch_size,
                i2v_mode, noise_mode, noise_strength, seed, 
                start_frames, frame_blend, wan_version, channel_mode, apply_wan_norm,
                qwen_latent=None, bridge_latent=None, start_image=None, 
                vae=None, clip_vision_output=None):
        
        torch.manual_seed(seed)
        
        info = []
        info.append("=== Unified I2V Bridge ===")
        info.append(f"Mode: {i2v_mode} | Noise: {noise_mode}")
        info.append(f"Dimensions: {width}x{height} | Frames: {num_frames}")
        
        # Determine input source and get latent
        input_latent = self._get_input_latent(
            qwen_latent, bridge_latent, start_image, vae, info
        )
        
        if input_latent is None:
            raise ValueError("No input provided! Connect qwen_latent, bridge_latent, or start_image")
        
        # Process based on WAN version
        channels = 16 if wan_version != "wan22" else 48
        info.append(f"Target channels: {channels}")
        
        # Prepare dimensions
        T = ((num_frames - 1) // 4) + 1
        target_H = height // 8
        target_W = width // 8
        
        # Process input latent
        processed_latent = self._process_latent(
            input_latent, target_H, target_W, channels, channel_mode, apply_wan_norm, info
        )
        
        # Create video latent based on mode
        if i2v_mode == "standard":
            video_latent, mask = self._create_standard_i2v(
                processed_latent, batch_size, T, target_H, target_W, 
                start_frames, frame_blend, info
            )
        elif i2v_mode == "direct_latent":
            video_latent, mask = self._create_direct_latent(
                processed_latent, batch_size, T, target_H, target_W,
                noise_mode, noise_strength, info
            )
        elif i2v_mode == "reference":
            video_latent, mask = self._create_reference_mode(
                processed_latent, batch_size, T, target_H, target_W,
                start_frames, info
            )
        elif i2v_mode == "hybrid":
            video_latent, mask = self._create_hybrid_mode(
                processed_latent, batch_size, T, target_H, target_W,
                frame_blend, noise_strength, info
            )
        elif i2v_mode == "vace_style":
            video_latent, mask = self._create_vace_style(
                processed_latent, batch_size, T, target_H, target_W,
                start_frames, noise_strength, info
            )
        
        # Apply noise modifications
        video_latent = self._apply_noise(
            video_latent, noise_mode, noise_strength, T, info
        )
        
        # Create conditioning
        concat_latent = video_latent.clone()
        
        # Apply conditioning to positive/negative
        positive_out = node_helpers.conditioning_set_values(
            positive, 
            {"concat_latent_image": concat_latent, "concat_mask": mask}
        )
        negative_out = node_helpers.conditioning_set_values(
            negative,
            {"concat_latent_image": concat_latent, "concat_mask": mask}
        )
        
        # Add clip vision if provided
        if clip_vision_output is not None:
            positive_out = node_helpers.conditioning_set_values(
                positive_out, {"clip_vision_output": clip_vision_output}
            )
            negative_out = node_helpers.conditioning_set_values(
                negative_out, {"clip_vision_output": clip_vision_output}
            )
            info.append("Added CLIP vision conditioning")
        
        # Create output latent (empty for generation)
        output_latent = torch.zeros(
            [batch_size, channels, T, target_H, target_W],
            device=comfy.model_management.intermediate_device()
        )
        
        out = {"samples": output_latent}
        info.append(f"Output shape: {output_latent.shape}")
        info.append("Ready for KSampler!")
        
        return (positive_out, negative_out, out, "\n".join(info))
    
    def _get_input_latent(self, qwen_latent, bridge_latent, start_image, vae, info):
        """Extract latent from whatever input is provided"""
        
        if qwen_latent is not None:
            info.append("Input: Qwen latent")
            latent = qwen_latent["samples"]
        elif bridge_latent is not None:
            info.append("Input: Bridge latent")
            latent = bridge_latent["samples"]
        elif start_image is not None and vae is not None:
            info.append("Input: Image (encoding with VAE)")
            # Ensure image is in right format
            if len(start_image.shape) == 3:
                start_image = start_image.unsqueeze(0)
            # Encode image
            latent = vae.encode(start_image[:, :, :, :3])
        else:
            return None
        
        # Normalize dimensions
        if len(latent.shape) == 5:
            # (B, C, F, H, W) - take first frame
            latent = latent[:, :, 0, :, :]
        elif len(latent.shape) == 3:
            # (C, H, W) - add batch
            latent = latent.unsqueeze(0)
        
        info.append(f"Input shape: {latent.shape}")
        return latent
    
    def _process_latent(self, latent, target_H, target_W, target_C, channel_mode, apply_norm, info):
        """Process latent to match target dimensions with WAN normalization"""
        
        B, C, H, W = latent.shape
        
        # Resize if needed
        if H != target_H or W != target_W:
            latent = F.interpolate(
                latent,
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=False
            )
            info.append(f"Resized: {H}x{W} → {target_H}x{target_W}")
        
        # Handle channel mismatch
        if C != target_C:
            if C < target_C:
                # Expand channels
                if target_C == 48 and C == 16:
                    if channel_mode == "frequency":
                        # Frequency-based expansion
                        high_freq = latent - F.avg_pool2d(F.avg_pool2d(latent, 3, 1, 1), 3, 1, 1)
                        low_freq = F.avg_pool2d(latent, 5, 1, 2)
                        latent = torch.cat([latent, high_freq, low_freq], dim=1)
                        info.append("Expanded: 16→48 channels (frequency-based)")
                    elif channel_mode == "repeat":
                        # Simple 3x repeat
                        latent = latent.repeat(1, 3, 1, 1)
                        info.append("Expanded: 16→48 channels (3x repeat)")
                    else:  # zero_pad
                        # Pad with zeros
                        padding = torch.zeros(B, 32, target_H, target_W, device=latent.device)
                        latent = torch.cat([latent, padding], dim=1)
                        info.append("Expanded: 16→48 channels (zero padding)")
                else:
                    # Generic padding
                    padding = torch.zeros(B, target_C - C, target_H, target_W, device=latent.device)
                    latent = torch.cat([latent, padding], dim=1)
                    info.append(f"Padded: {C}→{target_C} channels")
            else:
                # Truncate
                latent = latent[:, :target_C]
                info.append(f"Truncated: {C}→{target_C} channels")
        
        # Apply WAN normalization if requested
        if apply_norm:
            wan_mean = 0.0
            wan_std = 0.5
            current_mean = latent.mean()
            current_std = latent.std()
            
            if abs(current_mean - wan_mean) > 0.1 or abs(current_std - wan_std) > 0.1:
                latent = (latent - current_mean) / (current_std + 1e-8)
                latent = latent * wan_std + wan_mean
                info.append(f"Normalized: mean={wan_mean:.2f}, std={wan_std:.2f}")
        
        return latent
    
    def _create_standard_i2v(self, latent, batch_size, T, H, W, start_frames, blend, info):
        """Standard I2V conditioning"""
        
        # Create concat latent
        concat_latent = torch.zeros(
            [batch_size, latent.shape[1], T, H, W],
            device=latent.device,
            dtype=latent.dtype
        )
        
        # Fill start frames
        frames_to_fill = min(start_frames, T)
        for t in range(frames_to_fill):
            weight = blend * (1.0 - t / frames_to_fill) if blend < 1.0 else blend
            concat_latent[:, :, t] = latent[0] * weight
        
        # Create mask
        mask = torch.ones((1, 1, T, H, W), device=latent.device)
        mask[:, :, :frames_to_fill] = 0.0
        
        info.append(f"Standard I2V: {frames_to_fill} start frames")
        return concat_latent, mask
    
    def _create_direct_latent(self, latent, batch_size, T, H, W, noise_mode, strength, info):
        """Direct latent injection"""
        
        # Create video latent
        if noise_mode == "no_noise":
            video_latent = latent.unsqueeze(2).repeat(1, 1, T, 1, 1)
        else:
            # Start with noise
            video_latent = torch.randn(
                batch_size, latent.shape[1], T, H, W,
                device=latent.device
            ) * 0.1
            # Add latent influence
            for t in range(T):
                video_latent[:, :, t] += latent[0] * strength
        
        # No masking for direct mode
        mask = torch.zeros((1, 1, T, H, W), device=latent.device)
        
        info.append(f"Direct latent: strength={strength}")
        return video_latent, mask
    
    def _create_reference_mode(self, latent, batch_size, T, H, W, frames, info):
        """Reference/phantom mode"""
        
        # Mostly noise with reference frames
        video_latent = torch.randn(
            batch_size, latent.shape[1], T, H, W,
            device=latent.device
        ) * 0.1
        
        # Set reference frames
        ref_frames = min(frames, T)
        for t in range(ref_frames):
            video_latent[:, :, t] = latent[0]
        
        # Mask reference frames
        mask = torch.ones((1, 1, T, H, W), device=latent.device)
        mask[:, :, :ref_frames] = 0.0
        
        info.append(f"Reference mode: {ref_frames} frames")
        return video_latent, mask
    
    def _create_hybrid_mode(self, latent, batch_size, T, H, W, blend, strength, info):
        """Hybrid mode with decay"""
        
        video_latent = torch.zeros(
            batch_size, latent.shape[1], T, H, W,
            device=latent.device
        )
        
        for t in range(T):
            # Decay influence over time
            decay = blend * (1.0 - t / T)
            noise = torch.randn_like(latent[0]) * (1.0 - decay) * strength
            video_latent[:, :, t] = latent[0] * decay + noise
        
        # Gradual mask
        mask = torch.ones((1, 1, T, H, W), device=latent.device)
        for t in range(T):
            mask[:, :, t] *= (t / T)
        
        info.append(f"Hybrid mode: blend={blend}, strength={strength}")
        return video_latent, mask
    
    def _create_vace_style(self, latent, batch_size, T, H, W, keyframes, strength, info):
        """VACE-style keyframe conditioning"""
        
        video_latent = torch.zeros(
            batch_size, latent.shape[1], T, H, W,
            device=latent.device
        )
        
        # Set keyframes at regular intervals
        keyframe_interval = max(1, T // keyframes)
        for i in range(keyframes):
            t = min(i * keyframe_interval, T - 1)
            video_latent[:, :, t] = latent[0]
            
            # Interpolate between keyframes
            if i < keyframes - 1:
                next_t = min((i + 1) * keyframe_interval, T - 1)
                for inter_t in range(t + 1, next_t):
                    alpha = (inter_t - t) / (next_t - t)
                    noise = torch.randn_like(latent[0]) * strength
                    video_latent[:, :, inter_t] = latent[0] * (1 - alpha) + noise * alpha
        
        # Mask keyframes only
        mask = torch.ones((1, 1, T, H, W), device=latent.device)
        for i in range(keyframes):
            t = min(i * keyframe_interval, T - 1)
            mask[:, :, t] = 0.0
        
        info.append(f"VACE style: {keyframes} keyframes")
        return video_latent, mask
    
    def _apply_noise(self, latent, noise_mode, strength, T, info):
        """Apply noise based on mode"""
        
        if noise_mode == "no_noise":
            return latent
        
        device = latent.device
        shape = latent.shape
        
        if noise_mode == "add_noise":
            noise = torch.randn(shape, device=device) * strength
            latent = latent + noise
            info.append(f"Added noise: strength={strength}")
            
        elif noise_mode == "mix_noise":
            noise = torch.randn(shape, device=device)
            latent = latent * (1 - strength) + noise * strength
            info.append(f"Mixed noise: ratio={strength}")
            
        elif noise_mode == "scaled_noise":
            for t in range(T):
                t_strength = strength * (t / T)
                noise = torch.randn_like(latent[:, :, t]) * t_strength
                latent[:, :, t] = latent[:, :, t] + noise
            info.append(f"Scaled noise: max={strength}")
            
        elif noise_mode == "decay_noise":
            for t in range(T):
                t_strength = strength * (1.0 - t / T)
                noise = torch.randn_like(latent[:, :, t]) * t_strength
                latent[:, :, t] = latent[:, :, t] + noise
            info.append(f"Decay noise: initial={strength}")
        
        return latent