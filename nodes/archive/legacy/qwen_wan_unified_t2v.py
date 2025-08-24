"""
Unified T2V/V2V Bridge
Text-to-video and video-to-video generation with Qwen latent integration
"""

import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils
import node_helpers

class QwenWANUnifiedT2V:
    """
    Complete T2V/V2V solution with optional Qwen latent integration
    Handles text-to-video and video-to-video workflows
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
                "generation_mode": ([
                    "t2v",                # Pure text-to-video
                    "t2v_guided",         # T2V with Qwen guidance
                    "v2v",                # Video-to-video
                    "v2v_guided",         # V2V with Qwen guidance
                    "latent_morph",       # Morphing between latents
                ], {"default": "t2v"}),
                
                # Guidance control
                "guidance_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temporal_consistency": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Noise control for T2V
                "initial_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_schedule": ([
                    "constant",
                    "linear_decay", 
                    "exponential_decay",
                    "cosine",
                ], {"default": "constant"}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                
                # WAN version handling
                "wan_version": (["auto", "wan21", "wan22"], {"default": "auto"}),
                
                # Channel handling for WAN 2.2
                "channel_expansion": ([
                    "frequency",
                    "repeat",
                    "zero_pad",
                ], {"default": "frequency"}),
                
                # Normalization
                "apply_normalization": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Multiple input options
                "qwen_latent": ("LATENT",),          # For guidance
                "video_latent": ("LATENT",),         # For V2V
                "reference_latent": ("LATENT",),     # Additional reference
                "vae": ("VAE",),                     # For encoding if needed
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "info")
    FUNCTION = "process"
    CATEGORY = "QwenWANBridge/Unified"
    
    def process(self, positive, negative, width, height, num_frames, batch_size,
                generation_mode, guidance_strength, temporal_consistency,
                initial_noise, noise_schedule, seed, wan_version, 
                channel_expansion, apply_normalization,
                qwen_latent=None, video_latent=None, reference_latent=None,
                vae=None, clip_vision_output=None):
        
        torch.manual_seed(seed)
        
        info = []
        info.append("=== Unified T2V/V2V Bridge ===")
        info.append(f"Mode: {generation_mode}")
        info.append(f"Dimensions: {width}x{height} | Frames: {num_frames}")
        
        # Determine channels based on WAN version
        channels = 16 if wan_version != "wan22" else 48
        info.append(f"Target channels: {channels}")
        
        # Prepare dimensions
        T = ((num_frames - 1) // 4) + 1
        target_H = height // 8
        target_W = width // 8
        
        # Process based on generation mode
        if generation_mode == "t2v":
            latent, mask = self._create_t2v(
                batch_size, channels, T, target_H, target_W,
                initial_noise, noise_schedule, info
            )
            
        elif generation_mode == "t2v_guided":
            if qwen_latent is None:
                raise ValueError("t2v_guided requires qwen_latent input")
            latent, mask = self._create_t2v_guided(
                qwen_latent, batch_size, channels, T, target_H, target_W,
                guidance_strength, initial_noise, noise_schedule,
                channel_expansion, apply_normalization, info
            )
            
        elif generation_mode == "v2v":
            if video_latent is None:
                raise ValueError("v2v requires video_latent input")
            latent, mask = self._create_v2v(
                video_latent, channels, T, target_H, target_W,
                temporal_consistency, channel_expansion, apply_normalization, info
            )
            
        elif generation_mode == "v2v_guided":
            if video_latent is None:
                raise ValueError("v2v_guided requires video_latent input")
            latent, mask = self._create_v2v_guided(
                video_latent, qwen_latent, channels, T, target_H, target_W,
                guidance_strength, temporal_consistency,
                channel_expansion, apply_normalization, info
            )
            
        elif generation_mode == "latent_morph":
            latent, mask = self._create_latent_morph(
                qwen_latent, reference_latent, batch_size, channels, T, 
                target_H, target_W, channel_expansion, apply_normalization, info
            )
        
        # Apply conditioning
        positive_out = positive
        negative_out = negative
        
        # Add concat conditioning if we have a mask
        if mask is not None:
            positive_out = node_helpers.conditioning_set_values(
                positive, 
                {"concat_latent_image": latent.clone(), "concat_mask": mask}
            )
            negative_out = node_helpers.conditioning_set_values(
                negative,
                {"concat_latent_image": latent.clone(), "concat_mask": mask}
            )
            info.append("Added concat conditioning")
        
        # Add CLIP vision if provided
        if clip_vision_output is not None:
            positive_out = node_helpers.conditioning_set_values(
                positive_out, {"clip_vision_output": clip_vision_output}
            )
            negative_out = node_helpers.conditioning_set_values(
                negative_out, {"clip_vision_output": clip_vision_output}
            )
            info.append("Added CLIP vision conditioning")
        
        # Prepare output
        out = {"samples": latent}
        info.append(f"Output shape: {latent.shape}")
        
        return (positive_out, negative_out, out, "\n".join(info))
    
    def _create_t2v(self, batch_size, channels, T, H, W, initial_noise, schedule, info):
        """Pure text-to-video generation"""
        
        # Create noise latent
        latent = torch.randn(
            [batch_size, channels, T, H, W],
            device=comfy.model_management.intermediate_device()
        ) * initial_noise
        
        # Apply noise schedule
        if schedule != "constant":
            for t in range(T):
                if schedule == "linear_decay":
                    scale = 1.0 - (t / T)
                elif schedule == "exponential_decay":
                    scale = torch.exp(torch.tensor(-3.0 * t / T))
                elif schedule == "cosine":
                    scale = torch.cos(torch.tensor(3.14159 * t / (2 * T)))
                else:
                    scale = 1.0
                
                latent[:, :, t] *= scale.item()
        
        info.append(f"T2V: noise={initial_noise}, schedule={schedule}")
        
        # No mask for pure T2V
        return latent, None
    
    def _create_t2v_guided(self, qwen_latent, batch_size, channels, T, H, W,
                          guidance, initial_noise, schedule, channel_mode, 
                          apply_norm, info):
        """T2V with Qwen guidance"""
        
        # Extract and process Qwen latent
        qwen = self._extract_latent(qwen_latent, info)
        qwen = self._process_latent(qwen, H, W, channels, channel_mode, apply_norm, info)
        
        # Create base noise
        latent = torch.randn(
            [batch_size, channels, T, H, W],
            device=qwen.device
        ) * initial_noise
        
        # Apply Qwen guidance
        for t in range(T):
            # Decay guidance over time
            t_guidance = guidance * (1.0 - t / T)
            
            # Mix Qwen with noise
            latent[:, :, t] = latent[:, :, t] * (1 - t_guidance) + qwen[0] * t_guidance
        
        # Create mask for guided frames
        mask = torch.ones((1, 1, T, H, W), device=qwen.device)
        mask[:, :, 0] = 0.0  # First frame is guided
        
        info.append(f"T2V Guided: guidance={guidance}")
        
        return latent, mask
    
    def _create_v2v(self, video_latent, channels, T, H, W, consistency,
                   channel_mode, apply_norm, info):
        """Video-to-video transformation"""
        
        # Extract video latent
        video = self._extract_latent(video_latent, info)
        
        # Handle shape
        if len(video.shape) == 4:
            # Single frame, expand to video
            video = video.unsqueeze(2).repeat(1, 1, T, 1, 1)
        elif video.shape[2] != T:
            # Interpolate temporally
            video = F.interpolate(
                video,
                size=(T, H, W),
                mode='trilinear',
                align_corners=False
            )
        
        # Process channels and normalization
        video = self._process_video_latent(
            video, H, W, channels, channel_mode, apply_norm, info
        )
        
        # Add noise for variation
        noise = torch.randn_like(video) * (1.0 - consistency)
        video = video * consistency + noise
        
        info.append(f"V2V: consistency={consistency}")
        
        # No mask for V2V (full video transformation)
        return video, None
    
    def _create_v2v_guided(self, video_latent, qwen_latent, channels, T, H, W,
                          guidance, consistency, channel_mode, apply_norm, info):
        """V2V with Qwen guidance"""
        
        # Get base video
        video = self._extract_latent(video_latent, info)
        
        # Get Qwen guidance
        if qwen_latent is not None:
            qwen = self._extract_latent(qwen_latent, info)
            qwen = self._process_latent(qwen, H, W, channels, channel_mode, apply_norm, info)
        else:
            qwen = None
        
        # Process video
        if len(video.shape) == 4:
            video = video.unsqueeze(2).repeat(1, 1, T, 1, 1)
        elif video.shape[2] != T:
            video = F.interpolate(video, size=(T, H, W), mode='trilinear', align_corners=False)
        
        video = self._process_video_latent(
            video, H, W, channels, channel_mode, apply_norm, info
        )
        
        # Apply Qwen guidance if available
        if qwen is not None:
            for t in range(T):
                t_guidance = guidance * (1.0 - t / (2 * T))  # Slower decay for V2V
                video[:, :, t] = video[:, :, t] * (1 - t_guidance) + qwen[0] * t_guidance
        
        # Add controlled noise
        noise = torch.randn_like(video) * (1.0 - consistency)
        video = video * consistency + noise
        
        # Create mask for guided sections
        mask = torch.ones((1, 1, T, H, W), device=video.device)
        if qwen is not None:
            mask[:, :, 0] = 0.0  # First frame guided
        
        info.append(f"V2V Guided: guidance={guidance}, consistency={consistency}")
        
        return video, mask
    
    def _create_latent_morph(self, start_latent, end_latent, batch_size, channels, 
                            T, H, W, channel_mode, apply_norm, info):
        """Morph between two latents over time"""
        
        if start_latent is None and end_latent is None:
            # Pure noise if no inputs
            return self._create_t2v(batch_size, channels, T, H, W, 1.0, "constant", info)
        
        # Process start latent
        if start_latent is not None:
            start = self._extract_latent(start_latent, info)
            start = self._process_latent(start, H, W, channels, channel_mode, apply_norm, info)
        else:
            start = torch.randn(batch_size, channels, H, W, device=comfy.model_management.intermediate_device())
        
        # Process end latent
        if end_latent is not None:
            end = self._extract_latent(end_latent, info)
            end = self._process_latent(end, H, W, channels, channel_mode, apply_norm, info)
        else:
            end = torch.randn(batch_size, channels, H, W, device=start.device)
        
        # Create morphing sequence
        latent = torch.zeros([batch_size, channels, T, H, W], device=start.device)
        
        for t in range(T):
            alpha = t / (T - 1) if T > 1 else 0
            latent[:, :, t] = start[0] * (1 - alpha) + end[0] * alpha
        
        # Mask for keyframes
        mask = torch.ones((1, 1, T, H, W), device=start.device)
        mask[:, :, 0] = 0.0  # Start frame
        if T > 1:
            mask[:, :, -1] = 0.0  # End frame
        
        info.append(f"Latent Morph: {T} frames")
        
        return latent, mask
    
    def _extract_latent(self, latent_input, info):
        """Extract latent from input"""
        latent = latent_input["samples"]
        
        # Handle dimensions
        if len(latent.shape) == 5:
            # Video latent
            pass
        elif len(latent.shape) == 4:
            # Single frame
            pass
        elif len(latent.shape) == 3:
            # Add batch
            latent = latent.unsqueeze(0)
        
        return latent
    
    def _process_latent(self, latent, target_H, target_W, target_C, 
                       channel_mode, apply_norm, info):
        """Process single frame latent"""
        
        if len(latent.shape) == 5:
            latent = latent[:, :, 0]  # Take first frame
        
        B, C, H, W = latent.shape
        
        # Resize if needed
        if H != target_H or W != target_W:
            latent = F.interpolate(
                latent,
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=False
            )
            info.append(f"Resized: {H}x{W} to {target_H}x{target_W}")
        
        # Handle channels
        if C != target_C:
            latent = self._expand_channels(latent, target_C, channel_mode, info)
        
        # Normalize if requested
        if apply_norm:
            latent = self._normalize(latent, info)
        
        return latent
    
    def _process_video_latent(self, latent, target_H, target_W, target_C,
                             channel_mode, apply_norm, info):
        """Process video latent"""
        
        B, C, T, H, W = latent.shape
        
        # Resize spatially if needed
        if H != target_H or W != target_W:
            # Reshape for 2D interpolation
            latent = latent.view(B * T, C, H, W)
            latent = F.interpolate(
                latent,
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=False
            )
            latent = latent.view(B, C, T, target_H, target_W)
            info.append(f"Resized video: {H}x{W} to {target_H}x{target_W}")
        
        # Handle channels
        if C != target_C:
            # Process each frame
            frames = []
            for t in range(T):
                frame = self._expand_channels(latent[:, :, t], target_C, channel_mode, info)
                frames.append(frame)
            latent = torch.stack(frames, dim=2)
        
        # Normalize if requested
        if apply_norm:
            latent = self._normalize(latent, info)
        
        return latent
    
    def _expand_channels(self, latent, target_C, mode, info):
        """Expand channels from current to target"""
        B, C, *spatial = latent.shape
        
        if C >= target_C:
            return latent[:, :target_C]
        
        if target_C == 48 and C == 16:
            if mode == "frequency":
                # Frequency-based expansion
                high_freq = latent - F.avg_pool2d(F.avg_pool2d(latent.view(B, C, *spatial[-2:]), 3, 1, 1), 3, 1, 1).view(B, C, *spatial)
                low_freq = F.avg_pool2d(latent.view(B, C, *spatial[-2:]), 5, 1, 2).view(B, C, *spatial)
                latent = torch.cat([latent, high_freq, low_freq], dim=1)
                if info and "16 to 48 frequency" not in " ".join(info):
                    info.append("Expanded: 16 to 48 channels (frequency)")
            elif mode == "repeat":
                latent = latent.repeat(1, 3, *[1]*len(spatial))
                if info and "16 to 48 repeat" not in " ".join(info):
                    info.append("Expanded: 16 to 48 channels (repeat)")
            else:  # zero_pad
                padding_shape = [B, 32] + list(spatial)
                padding = torch.zeros(padding_shape, device=latent.device)
                latent = torch.cat([latent, padding], dim=1)
                if info and "16 to 48 zero" not in " ".join(info):
                    info.append("Expanded: 16 to 48 channels (zero pad)")
        else:
            # Generic padding
            padding_shape = [B, target_C - C] + list(spatial)
            padding = torch.zeros(padding_shape, device=latent.device)
            latent = torch.cat([latent, padding], dim=1)
            if info:
                info.append(f"Padded: {C} to {target_C} channels")
        
        return latent
    
    def _normalize(self, latent, info):
        """Apply WAN normalization"""
        wan_mean = 0.0
        wan_std = 0.5
        
        current_mean = latent.mean()
        current_std = latent.std()
        
        if abs(current_mean - wan_mean) > 0.1 or abs(current_std - wan_std) > 0.1:
            latent = (latent - current_mean) / (current_std + 1e-8)
            latent = latent * wan_std + wan_mean
            if info and "Normalized" not in " ".join(info):
                info.append(f"Normalized: mean={wan_mean:.2f}, std={wan_std:.2f}")
        
        return latent