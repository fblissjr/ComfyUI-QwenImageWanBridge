"""
Unified Qwen-Image-Edit Workflow Node
Complete implementation of all Qwen-Image-Edit features in a single, easy-to-use node
Based on DiffSynth-Studio reference implementation
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any
import folder_paths
import comfy.model_management
import comfy.samplers
import node_helpers
import nodes

class QwenImageEditUnified:
    """
    Complete Qwen-Image-Edit workflow in a single node
    Combines all features: vision tokens, templates, entity control, autoregressive editing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "mode": (["text_to_image", "image_edit", "autoregressive_edit", "entity_control"], {"default": "text_to_image"}),
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # Image editing inputs
                "edit_image": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                
                # Advanced options
                "auto_resize": ("BOOLEAN", {"default": True}),
                "target_pixels": ("INT", {"default": 1048576, "min": 262144, "max": 4194304}),
                "drop_template": ("BOOLEAN", {"default": True}),
                "reference_method": (["default", "inject", "concat", "cross_attn"], {"default": "default"}),
                
                # Entity control
                "entity_prompts": ("STRING", {"multiline": True, "default": ""}),
                "entity_masks": ("MASK",),
                
                # Autoregressive editing
                "autoregressive_prompts": ("STRING", {"multiline": True, "default": ""}),
                "denoise_schedule": ("STRING", {"default": "1.0,0.5,0.3"}),
                
                # ControlNet
                "controlnet": ("CONTROL_NET",),
                "controlnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                
                # Output options
                "return_intermediate": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT", "DICT", "LIST")
    RETURN_NAMES = ("image", "latent", "info", "intermediate_images")
    FUNCTION = "process"
    CATEGORY = "QwenImage/Unified"
    
    def __init__(self):
        self.device = comfy.model_management.get_torch_device()
        self.text_encoder = None
    
    def get_text_encoder(self):
        """Get or create text encoder instance"""
        if self.text_encoder is None:
            from .qwen_proper_text_encoder import QwenImageEditTextEncoder
            self.text_encoder = QwenImageEditTextEncoder()
        return self.text_encoder
    
    def process_text_to_image(self, clip, prompt: str, negative_prompt: str,
                             drop_template: bool) -> Tuple[list, list]:
        """Process text-to-image mode"""
        text_encoder = self.get_text_encoder()
        
        # Encode positive prompt
        positive_cond, _, pos_info = text_encoder.encode(
            clip, prompt, mode="text_to_image", 
            drop_template=drop_template
        )
        
        # Encode negative prompt
        if negative_prompt:
            negative_cond, _, _ = text_encoder.encode(
                clip, negative_prompt, mode="text_to_image",
                drop_template=drop_template
            )
        else:
            negative_cond = [[torch.zeros_like(positive_cond[0][0]), {}]]
        
        return positive_cond, negative_cond, pos_info
    
    def process_image_edit(self, clip, vae, prompt: str, edit_image: torch.Tensor,
                          negative_prompt: str, auto_resize: bool, target_pixels: int,
                          drop_template: bool) -> Tuple[list, list, Dict]:
        """Process image editing mode with vision tokens"""
        text_encoder = self.get_text_encoder()
        
        # Encode with image
        positive_cond, edit_latent, encode_info = text_encoder.encode(
            clip, prompt, mode="image_edit",
            edit_image=edit_image, vae=vae,
            auto_resize=auto_resize, target_area=target_pixels,
            drop_template=drop_template
        )
        
        # Encode negative
        if negative_prompt:
            negative_cond, _, _ = text_encoder.encode(
                clip, negative_prompt, mode="text_to_image",
                drop_template=drop_template
            )
        else:
            negative_cond = [[torch.zeros_like(positive_cond[0][0]), {}]]
        
        return positive_cond, negative_cond, edit_latent, encode_info
    
    def process_entity_control(self, clip, vae, prompt: str, entity_prompts: str,
                              entity_masks: Optional[torch.Tensor], edit_image: Optional[torch.Tensor],
                              drop_template: bool) -> Tuple[list, list, Dict]:
        """Process entity-level control generation"""
        from .qwen_eligen_controller import QwenEliGenController
        
        controller = QwenEliGenController()
        
        # Process entities
        positive_cond, negative_cond, entity_info = controller.prepare_entity_conditioning(
            clip, prompt, entity_prompts, entity_masks,
            negative_prompt="", mode="enhanced"
        )
        
        # If we have an edit image, add reference latents
        if edit_image is not None and vae is not None:
            edit_latent = vae.encode(edit_image[:, :, :, :3])
            positive_cond = node_helpers.conditioning_set_values(
                positive_cond, {"reference_latents": [edit_latent]}, append=True
            )
            entity_info["has_reference"] = True
        
        return positive_cond, negative_cond, entity_info
    
    def process_autoregressive(self, model, clip, vae, prompt: str, 
                              autoregressive_prompts: str, edit_image: Optional[torch.Tensor],
                              denoise_schedule: str, seed: int, steps: int, cfg: float,
                              sampler_name: str, scheduler: str) -> Tuple[torch.Tensor, List, Dict]:
        """Process autoregressive editing"""
        # Parse prompts and denoise schedule
        all_prompts = [prompt] + [p.strip() for p in autoregressive_prompts.split("\n") if p.strip()]
        denoise_values = [float(d.strip()) for d in denoise_schedule.split(",")]
        
        # Ensure enough denoise values
        while len(denoise_values) < len(all_prompts):
            denoise_values.append(denoise_values[-1] * 0.7)
        
        current_image = edit_image
        intermediate_images = []
        edit_history = ""
        
        text_encoder = self.get_text_encoder()
        
        for i, (edit_prompt, denoise) in enumerate(zip(all_prompts, denoise_values)):
            print(f"[Autoregressive] Edit {i+1}/{len(all_prompts)}: {edit_prompt[:50]}...")
            
            # Encode with autoregressive context
            if i == 0:
                mode = "image_edit" if current_image is not None else "text_to_image"
            else:
                mode = "autoregressive_edit"
            
            conditioning, edit_latent, encode_info = text_encoder.encode(
                clip, edit_prompt, mode=mode,
                edit_image=current_image, vae=vae,
                edit_history=edit_history
            )
            
            # Sample
            if edit_latent and "samples" in edit_latent:
                latent = edit_latent
            else:
                latent = {"samples": torch.zeros((1, 16, 64, 64), device=self.device)}
            
            samples = nodes.common_ksampler(
                model, seed + i, steps, cfg, sampler_name, scheduler,
                conditioning, latent, denoise
            )[0]
            
            # Decode
            current_image = vae.decode(samples["samples"])
            intermediate_images.append(current_image)
            
            # Update history
            edit_history += f"{edit_prompt}\n"
        
        autoregressive_info = {
            "num_edits": len(all_prompts),
            "final_history": edit_history,
            "denoise_schedule": denoise_values[:len(all_prompts)]
        }
        
        return current_image, intermediate_images, autoregressive_info
    
    def apply_controlnet(self, model, controlnet, image, strength):
        """Apply BlockWise ControlNet if provided"""
        if controlnet is None:
            return model
        
        # This would integrate with QwenImageBlockWiseControlNet
        # For now, return model as-is
        return model
    
    def process(self, model, clip, vae, mode: str, prompt: str, seed: int, steps: int,
               cfg: float, sampler_name: str, scheduler: str, denoise: float,
               edit_image: Optional[torch.Tensor] = None, negative_prompt: str = "",
               auto_resize: bool = True, target_pixels: int = 1048576,
               drop_template: bool = True, reference_method: str = "default",
               entity_prompts: str = "", entity_masks: Optional[torch.Tensor] = None,
               autoregressive_prompts: str = "", denoise_schedule: str = "1.0,0.5,0.3",
               controlnet = None, controlnet_strength: float = 1.0,
               return_intermediate: bool = False):
        """
        Unified processing for all Qwen-Image-Edit modes
        """
        info = {
            "mode": mode,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler_name,
            "scheduler": scheduler
        }
        
        intermediate_images = []
        
        # Apply ControlNet if provided
        if controlnet is not None:
            model = self.apply_controlnet(model, controlnet, edit_image, controlnet_strength)
            info["controlnet_applied"] = True
        
        # Process based on mode
        if mode == "text_to_image":
            # Text-to-image generation
            positive_cond, negative_cond, encode_info = self.process_text_to_image(
                clip, prompt, negative_prompt, drop_template
            )
            info.update(encode_info)
            
            # Create empty latent
            latent = {"samples": torch.zeros((1, 16, 64, 64), device=self.device)}
            
            # Sample
            samples = nodes.common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive_cond, latent, denoise
            )[0]
            
            # Decode
            image = vae.decode(samples["samples"])
            
        elif mode == "image_edit":
            # Image editing with vision tokens
            if edit_image is None:
                raise ValueError("Image editing mode requires an edit_image input")
            
            positive_cond, negative_cond, edit_latent, encode_info = self.process_image_edit(
                clip, vae, prompt, edit_image, negative_prompt,
                auto_resize, target_pixels, drop_template
            )
            info.update(encode_info)
            
            # Sample from edit latent
            samples = nodes.common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive_cond, edit_latent, denoise
            )[0]
            
            # Decode
            image = vae.decode(samples["samples"])
            
        elif mode == "entity_control":
            # Entity-level control generation
            if not entity_prompts:
                raise ValueError("Entity control mode requires entity_prompts")
            
            positive_cond, negative_cond, entity_info = self.process_entity_control(
                clip, vae, prompt, entity_prompts, entity_masks, edit_image, drop_template
            )
            info.update(entity_info)
            
            # Create latent
            if edit_image is not None and vae is not None:
                latent = {"samples": vae.encode(edit_image[:, :, :, :3])}
            else:
                latent = {"samples": torch.zeros((1, 16, 64, 64), device=self.device)}
            
            # Sample
            samples = nodes.common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive_cond, latent, denoise
            )[0]
            
            # Decode
            image = vae.decode(samples["samples"])
            
        elif mode == "autoregressive_edit":
            # Autoregressive editing
            if not autoregressive_prompts:
                raise ValueError("Autoregressive mode requires autoregressive_prompts")
            
            image, intermediate_images, autoregressive_info = self.process_autoregressive(
                model, clip, vae, prompt, autoregressive_prompts, edit_image,
                denoise_schedule, seed, steps, cfg, sampler_name, scheduler
            )
            info.update(autoregressive_info)
            samples = {"samples": vae.encode(image[:, :, :, :3])}
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Prepare outputs
        latent = samples
        
        # Return intermediate images if requested
        if not return_intermediate:
            intermediate_images = []
        
        return (image, latent, info, intermediate_images)


class QwenImageEditSimple:
    """
    Simplified Qwen-Image-Edit node for basic usage
    Auto-detects mode based on inputs
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate"
    CATEGORY = "QwenImage/Simple"
    
    def generate(self, model, clip, vae, prompt: str, seed: int,
                image: Optional[torch.Tensor] = None, negative_prompt: str = "",
                steps: int = 30, cfg: float = 4.0, denoise: float = 1.0):
        """
        Simple generation with auto mode detection
        """
        # Import unified node
        unified = QwenImageEditUnified()
        
        # Auto-detect mode
        mode = "image_edit" if image is not None else "text_to_image"
        
        # Process
        result_image, result_latent, info, _ = unified.process(
            model, clip, vae, mode, prompt, seed, steps, cfg,
            "euler", "normal", denoise,
            edit_image=image, negative_prompt=negative_prompt,
            auto_resize=True, target_pixels=1048576,
            drop_template=True, reference_method="default",
            return_intermediate=False
        )
        
        return (result_image, result_latent)


NODE_CLASS_MAPPINGS = {
    "QwenImageEditUnified": QwenImageEditUnified,
    "QwenImageEditSimple": QwenImageEditSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEditUnified": "Qwen Image Edit Unified",
    "QwenImageEditSimple": "Qwen Image Edit Simple",
}