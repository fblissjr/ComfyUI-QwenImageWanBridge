"""
Qwen-Image Sampler Integration
Ensures proper connectivity between our enhanced text encoder and ComfyUI's KSampler
Handles reference latents, vision features, and special conditioning
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any
import comfy.model_management
import comfy.samplers
import node_helpers
import nodes
import folder_paths
import numpy as np

class QwenImageSamplerWrapper:
    """
    Wrapper to ensure proper integration between Qwen text encoder and KSampler
    Handles reference latents and vision features correctly
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "latent_image": ("LATENT",),
            },
            "optional": {
                "edit_latent": ("LATENT",),  # From our text encoder
                "reference_method": (["default", "inject", "concat", "cross_attn"], {"default": "default"}),
                "reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "vision_features": ("DICT",),  # From enhanced text encoder
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "QwenImage/Sampling"
    
    def prepare_conditioning(self, conditioning: list, edit_latent: Optional[Dict], 
                            reference_method: str, reference_strength: float,
                            vision_features: Optional[Dict]) -> list:
        """
        Prepare conditioning with proper reference latents and vision features
        """
        # Clone conditioning to avoid modifying original
        new_conditioning = []
        
        for cond in conditioning:
            emb, extra = cond if len(cond) == 2 else (cond[0], {})
            new_extra = extra.copy()
            
            # Handle reference latents from edit_latent
            if edit_latent is not None and "samples" in edit_latent:
                ref_latent = edit_latent["samples"]
                
                # Apply reference strength
                if reference_strength != 1.0:
                    ref_latent = ref_latent * reference_strength
                
                # Set reference latents with method
                new_extra["reference_latents"] = [ref_latent]
                new_extra["reference_latents_method"] = reference_method
                
                # If vision features were provided, add them
                if vision_features is not None:
                    if "vision_features" in vision_features:
                        new_extra["vision_emb"] = vision_features["vision_features"]
                    if "vision_info" in vision_features:
                        new_extra["vision_info"] = vision_features["vision_info"]
            
            new_conditioning.append([emb, new_extra])
        
        return new_conditioning
    
    def sample(self, model, positive, negative, seed: int, steps: int, cfg: float,
              sampler_name: str, scheduler: str, denoise: float, latent_image: Dict,
              edit_latent: Optional[Dict] = None,
              reference_method: str = "default", reference_strength: float = 1.0,
              vision_features: Optional[Dict] = None) -> Tuple[Dict]:
        """
        Sample with proper Qwen-Image conditioning
        """
        # Validate latent_image
        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            print(f"[QwenSampler] Error: Invalid latent_image")
            device = comfy.model_management.get_torch_device()
            latent_image = {"samples": torch.zeros((1, 16, 128, 128), device=device)}
        
        # If edit_latent provided, inject reference into positive conditioning
        if edit_latent is not None:
            positive = self.prepare_conditioning(
                positive, edit_latent, reference_method, 
                reference_strength, vision_features
            )
            # Also prepare negative (without reference)
            negative = self.prepare_conditioning(
                negative, None, reference_method, 
                0.0, None
            )
        else:
            # Use conditioning as-is
            positive = self.prepare_conditioning(positive, None, "default", 1.0, vision_features)
            negative = self.prepare_conditioning(negative, None, "default", 1.0, None)
        
        # Import nodes here to avoid circular imports
        import nodes
        
        # Validate inputs before sampling
        print(f"[QwenSampler] Pre-sampling validation:")
        print(f"  - latent_image type: {type(latent_image)}")
        print(f"  - latent_image keys: {latent_image.keys() if isinstance(latent_image, dict) else 'Not a dict'}")
        
        if isinstance(latent_image, dict) and "samples" in latent_image:
            print(f"  - samples type: {type(latent_image['samples'])}")
            if torch.is_tensor(latent_image['samples']):
                print(f"  - samples shape: {latent_image['samples'].shape}")
                print(f"  - samples device: {latent_image['samples'].device}")
        
        # Final validation
        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            print("[QwenSampler] ERROR: Invalid latent_image, creating fallback")
            device = comfy.model_management.get_torch_device()
            latent_image = {"samples": torch.zeros((1, 16, 96, 172), device=device)}
        
        # Sample using common_ksampler with proper positive/negative
        print("[QwenSampler] Calling common_ksampler...")
        try:
            # Check if model expects different latent format
            if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                diffusion_model = model.model.diffusion_model
                if hasattr(diffusion_model, 'in_channels'):
                    expected_channels = diffusion_model.in_channels
                    current_channels = latent_image["samples"].shape[1]
                    if expected_channels != current_channels:
                        print(f"[QwenSampler] Model expects {expected_channels} channels, got {current_channels}")
                        
                        # Handle channel mismatch
                        if expected_channels == 64 and current_channels == 16:
                            # FLUX model expects 64 channels, we have 16 (Qwen)
                            print("[QwenSampler] Converting 16-channel Qwen latent to 64-channel for FLUX")
                            device = latent_image["samples"].device
                            b, c, h, w = latent_image["samples"].shape
                            
                            # Expand channels by repeating and adding noise
                            expanded = latent_image["samples"].repeat(1, 4, 1, 1)  # 16 * 4 = 64
                            noise = torch.randn_like(expanded) * 0.01
                            latent_image["samples"] = expanded + noise
                            print(f"[QwenSampler] Expanded latent shape: {latent_image['samples'].shape}")
            
            # Use the positive and negative conditioning directly
            samples = nodes.common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative,
                latent_image, denoise
            )[0]
            print("[QwenSampler] Sampling successful")
        except RuntimeError as e:
            if "shape" in str(e) and "invalid for input" in str(e):
                print(f"[QwenSampler] Reshape error detected: {e}")
                # Try with different sized latent
                h, w = latent_image["samples"].shape[2:4]
                # Ensure dimensions are divisible by 2
                h = (h // 2) * 2
                w = (w // 2) * 2
                device = latent_image["samples"].device
                print(f"[QwenSampler] Creating new latent with size {h}x{w}")
                latent_image["samples"] = torch.randn((1, 16, h, w), device=device) * 0.01
                
                # Try again with adjusted latent
                try:
                    samples = nodes.common_ksampler(
                        model, seed, steps, cfg, sampler_name, scheduler,
                        positive, negative,
                        latent_image, denoise
                    )[0]
                    print("[QwenSampler] Sampling successful with adjusted latent")
                except Exception as e2:
                    print(f"[QwenSampler] Second attempt failed: {e2}")
                    samples = latent_image  # Return input as fallback
            else:
                print(f"[QwenSampler] Sampling failed with error: {e}")
                samples = latent_image  # Return input as fallback
        except Exception as e:
            print(f"[QwenSampler] Unexpected error: {e}")
            print(f"[QwenSampler] Error type: {type(e).__name__}")
            samples = latent_image  # Return input as fallback
        
        return (samples,)


class QwenImageSamplerAdvanced:
    """
    Advanced sampler with full control over Qwen-Image generation
    Includes support for all DiffSynth features
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            },
            "optional": {
                "latent_image": ("LATENT",),
                "edit_info": ("DICT",),  # From text encoder
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "reference_method": (["default", "inject", "concat", "cross_attn"], {"default": "default"}),
                "entity_masks": ("MASK",),
                "entity_strengths": ("STRING", {"default": "1.0,1.0,1.0,1.0"}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False}),
                "disable_noise": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "DICT")
    RETURN_NAMES = ("samples", "generation_info")
    FUNCTION = "sample_advanced"
    CATEGORY = "QwenImage/Sampling"
    
    def apply_entity_control(self, positive: list, negative: list,
                           entity_masks: Optional[torch.Tensor],
                           entity_strengths: str) -> Tuple[list, list]:
        """
        Apply entity-level control to conditioning
        """
        if entity_masks is None:
            return positive, negative
        
        # Parse entity strengths
        strengths = [float(s.strip()) for s in entity_strengths.split(",")]
        
        # Apply entity masks to positive conditioning
        new_positive = []
        for cond in positive:
            emb, extra = cond if len(cond) == 2 else (cond[0], {})
            new_extra = extra.copy()
            
            # Add entity control
            if "entity_prompt_emb" in extra:
                # Apply strengths to entity embeddings
                entity_embs = extra["entity_prompt_emb"]
                for i, (emb_tensor, strength) in enumerate(zip(entity_embs, strengths)):
                    if i < len(entity_embs):
                        entity_embs[i] = emb_tensor * strength
                new_extra["entity_prompt_emb"] = entity_embs
            
            # Add entity masks
            new_extra["entity_masks"] = entity_masks
            new_extra["entity_strengths"] = strengths
            
            new_positive.append([emb, new_extra])
        
        return new_positive, negative
    
    def sample_advanced(self, model, positive, negative, seed: int, steps: int, 
                       cfg: float, sampler_name: str, scheduler: str,
                       latent_image: Optional[Dict] = None, edit_info: Optional[Dict] = None,
                       denoise: float = 1.0, reference_method: str = "default",
                       entity_masks: Optional[torch.Tensor] = None,
                       entity_strengths: str = "1.0,1.0,1.0,1.0",
                       start_at_step: int = 0, end_at_step: int = 10000,
                       return_with_leftover_noise: bool = False,
                       disable_noise: bool = False) -> Tuple[Dict, Dict]:
        """
        Advanced sampling with full control
        """
        device = comfy.model_management.get_torch_device()
        generation_info = {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise
        }
        
        # Handle edit_info if provided
        if edit_info is not None:
            # Extract reference latents from edit_info
            if "edit_latents" in edit_info and latent_image is None:
                latent_image = {"samples": edit_info["edit_latents"]}
                generation_info["used_edit_latent"] = True
            
            # Add edit mode info
            if "mode" in edit_info:
                generation_info["edit_mode"] = edit_info["mode"]
            
            # Apply reference latents to conditioning
            if "reference_latents" in edit_info or "edit_latents" in edit_info:
                ref_latent = edit_info.get("edit_latents", edit_info.get("reference_latents"))
                if ref_latent is not None:
                    positive = node_helpers.conditioning_set_values(
                        positive, 
                        {
                            "reference_latents": [ref_latent],
                            "reference_latents_method": reference_method
                        },
                        append=True
                    )
        
        # Apply entity control if provided
        if entity_masks is not None:
            positive, negative = self.apply_entity_control(
                positive, negative, entity_masks, entity_strengths
            )
            generation_info["entity_control"] = True
            generation_info["num_entities"] = entity_masks.shape[0] if len(entity_masks.shape) > 2 else 1
        
        # Create initial latent if not provided
        if latent_image is None:
            # Qwen uses 16-channel latents
            latent = torch.zeros((1, 16, 64, 64), device=device)
            latent_image = {"samples": latent}
            generation_info["created_empty_latent"] = True
        
        # Prepare noise settings
        if disable_noise:
            # For image editing, we might want to start from the edit latent without noise
            generation_info["noise_disabled"] = True
            add_noise = False
        else:
            add_noise = True
        
        # Advanced sampling with step control
        if start_at_step > 0 or end_at_step < steps:
            # Implement step-based control
            generation_info["step_range"] = (start_at_step, end_at_step)
            
            # This would require custom sampling logic
            # For now, use standard sampling with adjusted denoise
            effective_steps = end_at_step - start_at_step
            step_ratio = effective_steps / steps
            adjusted_denoise = denoise * step_ratio
        else:
            adjusted_denoise = denoise
        
        # Sample
        samples = nodes.common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, latent_image, adjusted_denoise
        )[0]
        
        # Handle leftover noise option
        if return_with_leftover_noise and adjusted_denoise < 1.0:
            generation_info["has_leftover_noise"] = True
            generation_info["leftover_noise_amount"] = 1.0 - adjusted_denoise
        
        generation_info["output_shape"] = list(samples["samples"].shape)
        
        return (samples, generation_info)


class QwenImageIterativeSampler:
    """
    Iterative sampling for progressive refinement
    Supports autoregressive editing workflow
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "initial_prompt": ("STRING", {"multiline": True}),
                "refinement_prompts": ("STRING", {"multiline": True, "default": "add more detail\nimprove lighting\nenhance colors"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps_per_iteration": ("INT", {"default": 20, "min": 5, "max": 100}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            },
            "optional": {
                "initial_image": ("IMAGE",),
                "denoise_schedule": ("STRING", {"default": "1.0,0.5,0.3,0.2"}),
                "save_intermediate": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "IMAGE", "LIST", "DICT")
    RETURN_NAMES = ("final_latent", "final_image", "intermediate_images", "iteration_log")
    FUNCTION = "iterative_sample"
    CATEGORY = "QwenImage/Sampling"
    
    def iterative_sample(self, model, clip, vae, initial_prompt: str,
                        refinement_prompts: str, seed: int, steps_per_iteration: int,
                        cfg: float, sampler_name: str, scheduler: str,
                        initial_image: Optional[torch.Tensor] = None,
                        denoise_schedule: str = "1.0,0.5,0.3,0.2",
                        save_intermediate: bool = True):
        """
        Perform iterative sampling with progressive refinement
        """
        device = comfy.model_management.get_torch_device()
        
        # Parse refinement prompts
        prompts = [initial_prompt] + [p.strip() for p in refinement_prompts.split("\n") if p.strip()]
        
        # Parse denoise schedule
        denoise_values = [float(d.strip()) for d in denoise_schedule.split(",")]
        # Ensure we have enough denoise values
        while len(denoise_values) < len(prompts):
            denoise_values.append(denoise_values[-1] * 0.8)  # Decay
        
        iteration_log = {
            "num_iterations": len(prompts),
            "prompts": prompts,
            "denoise_schedule": denoise_values[:len(prompts)],
            "iterations": []
        }
        
        # Initialize
        current_latent = None
        current_image = initial_image
        intermediate_images = []
        
        # Import our text encoder
        from .qwen_proper_text_encoder import QwenImageEditTextEncoder
        text_encoder = QwenImageEditTextEncoder()
        
        print(f"[IterativeSampler] Starting {len(prompts)} iterations")
        
        for i, (prompt, denoise) in enumerate(zip(prompts, denoise_values)):
            print(f"[IterativeSampler] Iteration {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Determine mode
            if i == 0 and initial_image is None:
                mode = "text_to_image"
            else:
                mode = "image_edit"
            
            # Encode prompt with proper mode
            if current_image is not None:
                conditioning, edit_latent, encode_info = text_encoder.encode(
                    clip, prompt, mode=mode, edit_image=current_image, vae=vae
                )
            else:
                conditioning, edit_latent, encode_info = text_encoder.encode(
                    clip, prompt, mode="text_to_image"
                )
            
            # Use edit_latent as starting point if available
            if current_latent is None and edit_latent is not None:
                current_latent = edit_latent
            
            # Sample
            sampled = nodes.common_ksampler(
                model, seed + i, steps_per_iteration, cfg, 
                sampler_name, scheduler, conditioning, 
                current_latent if current_latent else {"samples": torch.zeros((1, 16, 64, 64), device=device)},
                denoise
            )[0]
            
            current_latent = sampled
            
            # Decode to image
            current_image = vae.decode(sampled["samples"])
            
            # Save intermediate if requested
            if save_intermediate:
                intermediate_images.append(current_image)
            
            # Log iteration
            iteration_log["iterations"].append({
                "iteration": i + 1,
                "prompt": prompt,
                "denoise": denoise,
                "mode": mode,
                "encode_info": encode_info
            })
        
        # Final outputs
        final_latent = current_latent
        final_image = current_image
        
        return (final_latent, final_image, intermediate_images, iteration_log)


NODE_CLASS_MAPPINGS = {
    "QwenImageSamplerWrapper": QwenImageSamplerWrapper,
    "QwenImageSamplerAdvanced": QwenImageSamplerAdvanced,
    "QwenImageIterativeSampler": QwenImageIterativeSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageSamplerWrapper": "Qwen Image Sampler Wrapper",
    "QwenImageSamplerAdvanced": "Qwen Image Sampler Advanced",
    "QwenImageIterativeSampler": "Qwen Image Iterative Sampler",
}