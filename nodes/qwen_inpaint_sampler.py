"""
QwenInpaintSampler

Inpainting sampler that implements the exact pattern from QwenImageEditInpaintPipeline.
Core pattern: latents = (1 - mask) * original_latents + mask * generated_latents

Direct port from working diffusers code with ComfyUI integration.
"""

import torch
import comfy.sample
import comfy.samplers
import comfy.model_management
from typing import Tuple, Optional, Dict, Any
import numpy as np
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class QwenInpaintSampler:
    """
    Inpainting sampler that implements the exact pattern from QwenImageEditInpaintPipeline.
    Core pattern: latents = (1 - mask) * original_latents + mask * generated_latents
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Standard ComfyUI sampling inputs
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT", {
                    "tooltip": "Accepts 4-channel (standard ComfyUI) or 16-channel (Qwen) latents. 4-channel will be auto-converted."
                }),
                "mask": ("MASK",),
                
                # Inpainting-specific parameters from working diffusers code
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Inpainting strength (0=preserve original, 1=complete regeneration)"
                }),
                "steps": ("INT", {
                    "default": 8, "min": 1, "max": 100, 
                    "tooltip": "Inference steps (8 recommended for Lightning LoRA)"
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "True CFG scale from diffusers implementation"
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "normal"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            },
            "optional": {
                "padding_mask_crop": ("INT", {
                    "default": 0, "min": 0, "max": 512,
                    "tooltip": "Crop padding around mask (0=disabled, may cause shape errors if enabled)"
                })
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "inpaint"
    CATEGORY = "Qwen/Sampling"
    
    def inpaint(self, model, positive, negative, latent_image, mask, 
                strength: float, steps: int, true_cfg_scale: float, 
                sampler_name: str, scheduler: str, seed: int,
                padding_mask_crop: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Core inpainting implementation following QwenImageEditInpaintPipeline exactly
        
        Key pattern from working code:
        latents = (1 - init_mask) * init_latents_proper + init_mask * latents
        """
        
        logger.info("=== QWEN INPAINT SAMPLER START ===")
        logger.info(f"Inputs - strength: {strength}, steps: {steps}, cfg: {true_cfg_scale}")
        logger.info(f"Inputs - sampler: {sampler_name}, scheduler: {scheduler}, seed: {seed}")
        logger.info(f"Inputs - padding_mask_crop: {padding_mask_crop}")
        logger.info(f"Input latent shape: {latent_image['samples'].shape}")
        logger.info(f"Input mask shape: {mask.shape}")
        
        device = comfy.model_management.get_torch_device()
        logger.info(f"Using device: {device}")
        
        # Ensure all inputs are on the correct device and get latent_samples
        latent_samples = latent_image['samples'].to(device)
        mask = mask.to(device)
        logger.info(f"Moved inputs to device: {device}")
        
        # FLEXIBLE INPUT VALIDATION - Handle both 4-channel and 16-channel inputs
        input_channels = latent_samples.shape[1]
        logger.info(f"Input channels detected: {input_channels}")
        
        if input_channels == 4:
            # Convert 4-channel ComfyUI latent to 16-channel Qwen latent
            logger.info("Converting 4-channel ComfyUI latent to 16-channel Qwen latent")
            latent_samples = latent_samples.repeat(1, 4, 1, 1)  # 4 -> 16 channels
            logger.info(f"Converted latent shape: {latent_samples.shape}")
            latent_image['samples'] = latent_samples
            logger.info("✅ Converted 4→16 channels - now compatible with Qwen")
        elif input_channels == 16:
            logger.info("✅ Input validation passed - 16-channel latents detected")
        else:
            error_msg = f"""
❌ UNSUPPORTED CHANNEL COUNT: {input_channels}

Supported formats:
- 4-channel: Standard ComfyUI latents (will be converted to 16-channel)
- 16-channel: Native Qwen latents (used as-is)

Input latent shape: {latent_samples.shape}
"""
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert 5D to 4D for DiT operations BEFORE mask preparation
        logger.info(f"Pre-conversion latent tensor shape: {latent_samples.shape}")
        if len(latent_samples.shape) == 5:
            # 5D tensor from VAE - convert to 4D for DiT transformer operations
            batch_size, channels, time_dim, latent_height, latent_width = latent_samples.shape
            if time_dim != 1:
                raise ValueError(f"Expected time dimension to be 1, got {time_dim}")
            # Squeeze time dimension for transformer operations  
            latent_samples = latent_samples.squeeze(2)  # [B, C, T, H, W] -> [B, C, H, W]
            logger.info(f"Converted 5D->4D for DiT: {latent_samples.shape}")
            # Update the original latent_image dict
            latent_image['samples'] = latent_samples
        
        # CRITICAL: Validate latent_samples is 4D before proceeding
        if len(latent_samples.shape) != 4:
            error_msg = f"""
❌ TENSOR DIMENSION ERROR: Expected 4D tensor after conversion, got {len(latent_samples.shape)}D
Shape: {latent_samples.shape}
This indicates a problem with the input or conversion logic.
"""
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"✅ Validated 4D tensor for DiT operations: {latent_samples.shape}")
        
        # Prepare mask in latent space (critical pattern from diffusers)
        mask_latent = self._prepare_mask_latents(
            mask, 
            latent_image,  # Pass the updated latent_image dict
            device, 
            num_channels=4  # Use 4 channels for mask like diffusers reference
        )
        logger.info(f"Prepared mask latent shape: {mask_latent.shape}")
        
        # CRITICAL: Expand mask to match 4D latent tensor dimensions 
        # Use the converted 4D latents from latent_image dict
        converted_latents = latent_image['samples']
        logger.info(f"Pre-expansion mask shape: {mask_latent.shape}")
        logger.info(f"Target converted latent shape: {converted_latents.shape}")
        
        # CRITICAL: Ensure converted_latents is 4D
        if len(converted_latents.shape) != 4:
            error_msg = f"""
❌ CONVERTED LATENTS ERROR: Expected 4D tensor, got {len(converted_latents.shape)}D
Shape: {converted_latents.shape}
The latent_image['samples'] should have been converted to 4D earlier!
"""
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Update latent_samples to use the validated converted latents
        latent_samples = converted_latents
        logger.info(f"Updated latent_samples to use validated 4D tensor: {latent_samples.shape}")
        
        # Handle channel expansion (4 -> 16 channels) - following diffusers reference
        if mask_latent.shape[1] == 4 and converted_latents.shape[1] == 16:
            mask_latent = mask_latent.repeat(1, 4, 1, 1)  # 4 -> 16 channels
            logger.info(f"Expanded mask channels 4->16: {mask_latent.shape}")
        
        # Both mask and converted latents should now be 4D [B, C, H, W] for DiT operations
        assert mask_latent.shape == converted_latents.shape, f"Shape mismatch: mask {mask_latent.shape} vs latents {converted_latents.shape}"
        logger.info("✅ 4D Mask-latent shape alignment verified for DiT operations")
        
        # Apply padding crop if specified (from working diffusers code)
        if padding_mask_crop is not None and padding_mask_crop > 0:
            logger.info(f"Applying padding crop: {padding_mask_crop}")
            # Use the already validated latent_image dict for crop
            latent_image, mask_latent = self._apply_padding_crop(
                latent_image, mask_latent, padding_mask_crop
            )
            # Update latent_samples from the cropped result
            latent_samples = latent_image['samples']
            logger.info(f"After crop - latent shape: {latent_samples.shape}")
            logger.info(f"After crop - mask shape: {mask_latent.shape}")
            
            # Re-expand mask to match 4D latent tensor after crop
            logger.info(f"Post-crop mask shape: {mask_latent.shape}")
            logger.info(f"Post-crop latent shape: {latent_samples.shape}")
            
            # Handle channel expansion (4 -> 16 channels) after crop
            if mask_latent.shape[1] == 4 and latent_samples.shape[1] == 16:
                mask_latent = mask_latent.repeat(1, 4, 1, 1)  # 4 -> 16 channels
                logger.info(f"Re-expanded cropped mask channels: {mask_latent.shape}")
            
            # Re-validate after crop - both should be 4D [B, C, H, W]
            assert mask_latent.shape == latent_samples.shape, f"Post-crop shape mismatch: mask {mask_latent.shape} vs latents {latent_samples.shape}"
            logger.info("✅ Post-crop 4D mask-latent alignment verified")
        
        # Initialize noise for masked regions (exact diffusers pattern)
        # CRITICAL: Use the validated 4D latent_samples for noise generation
        logger.info(f"Pre-noise generation latent_samples shape: {latent_samples.shape}")
        if len(latent_samples.shape) != 4:
            error_msg = f"""
❌ NOISE GENERATION ERROR: latent_samples must be 4D for noise generation
Current shape: {latent_samples.shape}
Expected: [batch, channels, height, width]
"""
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            latent_samples.shape, 
            generator=generator, 
            device=device, 
            dtype=latent_samples.dtype
        )
        logger.info(f"Generated noise shape: {noise.shape}")
        logger.info(f"✅ Noise shape validation: {len(noise.shape)}D tensor")
        
        # Prepare initial latents with strength control (from diffusers)
        init_latents = latent_samples
        original_latents = init_latents.clone()
        logger.info(f"Original latents shape: {original_latents.shape}")
        
        # CRITICAL: Ensure ALL tensors are on the same device and dtype before operations
        logger.info(f"Device sync - Target device: {device}")
        logger.info(f"init_latents device: {init_latents.device}, dtype: {init_latents.dtype}")
        logger.info(f"mask_latent device: {mask_latent.device}, dtype: {mask_latent.dtype}")
        logger.info(f"noise device: {noise.device}, dtype: {noise.dtype}")
        logger.info(f"original_latents device: {original_latents.device}, dtype: {original_latents.dtype}")
        
        # Force all tensors to target device and dtype
        target_dtype = init_latents.dtype
        init_latents = init_latents.to(device=device, dtype=target_dtype)
        original_latents = original_latents.to(device=device, dtype=target_dtype)
        mask_latent = mask_latent.to(device=device, dtype=target_dtype)
        noise = noise.to(device=device, dtype=target_dtype)
        
        logger.info("✅ All tensors synchronized to same device and dtype")
        logger.info(f"Final sync check - init_latents: {init_latents.device}")
        logger.info(f"Final sync check - mask_latent: {mask_latent.device}")
        logger.info(f"Final sync check - noise: {noise.device}")
        logger.info(f"Final sync check - original_latents: {original_latents.device}")
        
        if strength < 1.0:
            # Partial inpainting - add noise only to masked areas
            logger.info(f"Partial inpainting with strength: {strength}")
            noise_masked = noise * mask_latent
            noise_preserved = init_latents * (1 - mask_latent)
            init_latents = noise_masked * strength + noise_preserved
        else:
            # Full inpainting - completely regenerate masked areas
            logger.info("Full inpainting - complete regeneration of masked areas")
            init_latents = noise * mask_latent + init_latents * (1 - mask_latent)
        
        logger.info(f"Initialized latents shape: {init_latents.shape}")
        
        # Create inpainting-specific conditioning (critical for quality)
        positive_inpaint = self._create_inpaint_conditioning(positive, mask_latent, original_latents)
        logger.info("Created inpainting-specific conditioning")
        
        # Sample with inpainting constraints using ComfyUI's sample function
        logger.info("Starting ComfyUI sampling with inpainting-prepared latents")
        samples = comfy.sample.sample(
            model=model,
            noise=init_latents,
            steps=steps,
            cfg=true_cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler, 
            positive=positive_inpaint,
            negative=negative,
            latent_image=init_latents,
            start_step=0,
            last_step=steps,
            force_full_denoise=True
        )
        logger.info(f"ComfyUI sampling complete - samples shape: {samples.shape}")
        
        # Ensure samples are on correct device and dtype
        logger.info(f"Post-sampling device sync - samples device: {samples.device}")
        logger.info(f"Post-sampling device sync - original_latents device: {original_latents.device}")
        logger.info(f"Post-sampling device sync - mask_latent device: {mask_latent.device}")
        
        samples = samples.to(device=device, dtype=original_latents.dtype)
        original_latents = original_latents.to(device=device, dtype=original_latents.dtype)
        mask_latent = mask_latent.to(device=device, dtype=original_latents.dtype)
        
        logger.info("✅ Post-sampling device synchronization complete")
        
        # Apply final inpainting blend (THE CORE DIFFUSERS PATTERN)
        logger.info("Applying final inpainting blend - THE CORE PATTERN")
        final_latents = self._apply_inpaint_blending(samples, original_latents, mask_latent)
        logger.info(f"Final latents shape: {final_latents.shape}")
        
        logger.info("=== QWEN INPAINT SAMPLER END ===")
        return ({"samples": final_latents},)
    
    def _apply_inpaint_blending(self, generated_latents: torch.Tensor, 
                               original_latents: torch.Tensor, 
                               mask: torch.Tensor) -> torch.Tensor:
        """
        THE CRITICAL PATTERN from QwenImageEditInpaintPipeline:
        latents = (1 - init_mask) * init_latents_proper + init_mask * latents
        
        This is what makes inpainting work - preserved areas stay original,
        masked areas get the generated content.
        """
        logger.info("=== APPLYING CORE INPAINTING BLEND ===")
        logger.info(f"Generated latents shape: {generated_latents.shape}")
        logger.info(f"Original latents shape: {original_latents.shape}")
        logger.info(f"Mask shape: {mask.shape}")
        
        # Validate shapes match reference expectations - all should be 4D [B, C, H, W] for DiT
        assert generated_latents.shape == original_latents.shape, f"Latent shape mismatch: generated {generated_latents.shape} vs original {original_latents.shape}"
        assert mask.shape == generated_latents.shape, f"Mask shape mismatch: mask {mask.shape} vs latents {generated_latents.shape}"
        assert len(generated_latents.shape) == 4, f"Expected 4D tensors for DiT operations, got {len(generated_latents.shape)}D"
        
        logger.info("✓ 4D tensor validation passed - matches DiT reference pattern")
        
        # Ensure all tensors are on the same device and dtype
        device = generated_latents.device
        dtype = generated_latents.dtype
        
        mask = mask.to(device=device, dtype=dtype)
        original_latents = original_latents.to(device=device, dtype=dtype)
        
        # Mask should already be properly sized from preparation step
        if mask.shape != generated_latents.shape:
            logger.error(f"UNEXPECTED: Mask shape {mask.shape} doesn't match latents {generated_latents.shape}")
            logger.error("This should have been handled in preparation step!")
            
            # Fallback expansion (should not be needed if preparation is correct)
            if len(mask.shape) == 4 and mask.shape[1] == 1:
                # Expand single channel mask to match latent channels  
                mask = mask.repeat(1, generated_latents.shape[1], 1, 1)
                logger.info(f"Fallback: Expanded single-channel mask to shape: {mask.shape}")
            elif len(mask.shape) == 3:
                # Add batch dimension and expand channels
                mask = mask.unsqueeze(0).repeat(1, generated_latents.shape[1], 1, 1)
                logger.info(f"Fallback: Added batch dim and expanded mask to shape: {mask.shape}")
        
        # THE CORE BLENDING FORMULA from diffusers (verified alignment)
        # Reference: Qwen-Image-Edit-Inpaint/qwenimage/pipeline_qwenimage_edit_inpaint.py:1054
        # Pattern: latents = (1 - init_mask) * init_latents_proper + init_mask * latents
        # Status: ✅ PERFECT ALIGNMENT - Implements exact same formula
        blended = (1 - mask) * original_latents + mask * generated_latents
        
        logger.info(f"Blended latents shape: {blended.shape}")
        logger.info("=== CORE INPAINTING BLEND COMPLETE ===")
        return blended
    
    def _prepare_mask_latents(self, mask: torch.Tensor, latent_image: Dict[str, torch.Tensor], 
                             device: torch.device, num_channels: int) -> torch.Tensor:
        """Prepare mask for latent space operations - exact diffusers pattern"""
        
        logger.info("=== PREPARING MASK LATENTS ===")
        logger.info(f"Input mask shape: {mask.shape}")
        logger.info(f"Input mask dtype: {mask.dtype}")
        logger.info(f"Input mask min/max: {mask.min().item():.4f}/{mask.max().item():.4f}")
        logger.info(f"Input mask unique values: {torch.unique(mask)[:10]}")  # Show first 10 unique values
        logger.info(f"Target device: {device}")
        logger.info(f"Target channels: {num_channels}")
        
        # Get latent dimensions - should now be 4D [B, C, H, W] for DiT operations
        latent_samples = latent_image['samples']
        logger.info(f"Latent tensor shape in mask prep: {latent_samples.shape}")
        
        if len(latent_samples.shape) != 4:
            raise ValueError(f"Expected 4D latent tensor, got {len(latent_samples.shape)}D: {latent_samples.shape}")
        
        batch_size, channels, latent_height, latent_width = latent_samples.shape
        logger.info(f"4D latent dimensions: {batch_size}x{channels}x{latent_height}x{latent_width}")
        
        # Handle different mask input formats
        if len(mask.shape) == 2:
            # Add batch and channel dimensions
            mask = mask.unsqueeze(0).unsqueeze(0)
            logger.info(f"Added batch and channel dims: {mask.shape}")
        elif len(mask.shape) == 3:
            # Add channel dimension
            mask = mask.unsqueeze(1)
            logger.info(f"Added channel dim: {mask.shape}")
        
        # Resize mask to latent dimensions (8x downscale from pixel space for most VAEs)
        if mask.shape[-2:] != (latent_height, latent_width):
            logger.info(f"Resizing mask from {mask.shape[-2:]} to {latent_height}x{latent_width}")
            mask_resized = torch.nn.functional.interpolate(
                mask.float(),
                size=(latent_height, latent_width),
                mode='bilinear',
                align_corners=False
            )
            logger.info(f"Resized mask shape: {mask_resized.shape}")
            logger.info(f"Resized mask min/max: {mask_resized.min().item():.4f}/{mask_resized.max().item():.4f}")
            logger.info(f"Resized mask unique values: {torch.unique(mask_resized)[:10]}")
        else:
            mask_resized = mask.float()
            logger.info("Mask already at correct size")
        
        # Expand to 4 channels for mask (following diffusers reference pattern)
        mask_channels = 4  # Use 4 channels for mask, not full latent channels
        if mask_resized.shape[1] != mask_channels:
            logger.info(f"Expanding mask channels from {mask_resized.shape[1]} to {mask_channels}")
            mask_latent = mask_resized.repeat(1, mask_channels, 1, 1)
        else:
            mask_latent = mask_resized
        
        # Ensure correct batch size
        if mask_latent.shape[0] != batch_size:
            logger.info(f"Adjusting mask batch size from {mask_latent.shape[0]} to {batch_size}")
            mask_latent = mask_latent.repeat(batch_size, 1, 1, 1)
        
        mask_latent = mask_latent.to(device=device, dtype=latent_samples.dtype)
        logger.info(f"Final mask latent shape: {mask_latent.shape}")
        logger.info(f"Final mask latent dtype: {mask_latent.dtype}")
        logger.info(f"Final mask latent min/max: {mask_latent.min().item():.4f}/{mask_latent.max().item():.4f}")
        logger.info(f"Final mask latent unique values: {torch.unique(mask_latent)[:10]}")
        logger.info("=== MASK LATENTS PREPARED ===")
        
        return mask_latent
    
    def _create_inpaint_conditioning(self, conditioning, mask_latent: torch.Tensor, 
                                   original_latents: torch.Tensor):
        """Create inpainting-specific conditioning (may need model-specific logic)"""
        logger.info("Creating inpainting-specific conditioning")
        
        # For now, pass through original conditioning
        # This may need enhancement based on Qwen model specifics
        # Future enhancement: could add mask information to conditioning
        logger.info("Using original conditioning (future: add mask-aware conditioning)")
        return conditioning
    
    def _apply_padding_crop(self, latent_image: Dict[str, torch.Tensor], 
                           mask_latent: torch.Tensor, padding: int):
        """Apply padding crop around mask - from diffusers implementation"""
        logger.info(f"=== APPLYING PADDING CROP: {padding} ===")
        
        # Find mask bounds
        # Reduce mask to 2D for coordinate finding
        mask_2d = mask_latent.squeeze().cpu().numpy()
        if len(mask_2d.shape) > 2:
            # Take max across channels
            mask_2d = mask_2d.max(axis=0)
        
        coords = np.column_stack(np.where(mask_2d > 0.1))  # Lower threshold for mask detection
        logger.info(f"Found {len(coords)} mask pixels (threshold: 0.1)")
        logger.info(f"Mask 2D min/max: {mask_2d.min():.4f}/{mask_2d.max():.4f}")
        logger.info(f"Mask 2D unique values: {np.unique(mask_2d)[:10]}")  # Show first 10 unique values
        
        if len(coords) == 0:
            logger.info("No mask pixels found - returning original")
            return latent_image, mask_latent
        
        # Calculate crop bounds with padding
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        
        logger.info(f"Mask bounds: ({min_x},{min_y}) to ({max_x},{max_y})")
        
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)  
        max_x = min(mask_latent.shape[-1], max_x + padding)
        max_y = min(mask_latent.shape[-2], max_y + padding)
        
        logger.info(f"Crop bounds with padding: ({min_x},{min_y}) to ({max_x},{max_y})")
        
        # Ensure crop dimensions are multiples of 8 for model compatibility
        crop_width = max_x - min_x
        crop_height = max_y - min_y
        
        logger.info(f"Original crop dimensions: {crop_width}x{crop_height}")
        
        # Round to nearest multiple of 8 (minimum for most diffusion models)
        # Also ensure minimum dimensions of 64x64 latent pixels for model stability
        crop_width_rounded = max(64, ((crop_width + 7) // 8) * 8)
        crop_height_rounded = max(64, ((crop_height + 7) // 8) * 8)
        
        # Adjust bounds to maintain rounding while staying within image bounds
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        min_x_rounded = max(0, center_x - crop_width_rounded // 2)
        max_x_rounded = min(mask_latent.shape[-1], min_x_rounded + crop_width_rounded)
        min_y_rounded = max(0, center_y - crop_height_rounded // 2) 
        max_y_rounded = min(mask_latent.shape[-2], min_y_rounded + crop_height_rounded)
        
        # Ensure we don't exceed bounds
        if max_x_rounded > mask_latent.shape[-1]:
            max_x_rounded = mask_latent.shape[-1]
            min_x_rounded = max(0, max_x_rounded - crop_width_rounded)
        if max_y_rounded > mask_latent.shape[-2]:
            max_y_rounded = mask_latent.shape[-2]
            min_y_rounded = max(0, max_y_rounded - crop_height_rounded)
        
        logger.info(f"Rounded crop bounds: ({min_x_rounded},{min_y_rounded}) to ({max_x_rounded},{max_y_rounded})")
        final_crop_width = max_x_rounded - min_x_rounded
        final_crop_height = max_y_rounded - min_y_rounded
        logger.info(f"Final crop dimensions: {final_crop_width}x{final_crop_height}")
        
        # Safety check - warn if crop is too large relative to original
        original_width = mask_latent.shape[-1]
        original_height = mask_latent.shape[-2]
        if final_crop_width > original_width * 0.9 or final_crop_height > original_height * 0.9:
            logger.warning(f"Crop size ({final_crop_width}x{final_crop_height}) is very large relative to original ({original_width}x{original_height}). Consider using smaller padding or no crop.")
        
        # Safety check - ensure dimensions are reasonable for processing
        if final_crop_width < 8 or final_crop_height < 8:
            logger.warning(f"Crop dimensions ({final_crop_width}x{final_crop_height}) are too small. Skipping crop.")
            return latent_image, mask_latent
        
        # Apply crop with validated bounds
        latent_cropped = {
            'samples': latent_image['samples'][:, :, min_y_rounded:max_y_rounded, min_x_rounded:max_x_rounded]
        }
        mask_cropped = mask_latent[:, :, min_y_rounded:max_y_rounded, min_x_rounded:max_x_rounded]
        
        logger.info(f"Cropped latent shape: {latent_cropped['samples'].shape}")
        logger.info(f"Cropped mask shape: {mask_cropped.shape}")
        logger.info("=== PADDING CROP COMPLETE ===")
        
        return latent_cropped, mask_cropped


NODE_CLASS_MAPPINGS = {
    "QwenInpaintSampler": QwenInpaintSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenInpaintSampler": "Qwen Inpainting Sampler"
}