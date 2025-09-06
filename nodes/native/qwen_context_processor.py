"""
QwenContextProcessor - Context image handling for ControlNet-style workflows

This module provides DiffSynth-style context image processing that enables
ControlNet workflows with native Qwen models. Context images are processed
separately from vision tokens to provide spatial conditioning without
interfering with the text-to-vision understanding.

Reference implementations:
- DiffSynth-Engine: latent concatenation approach
- DiffSynth-Studio: conditioning integration patterns
"""

import torch
import logging
from typing import Tuple, Optional, Dict, Any, List
from PIL import Image
import numpy as np

import comfy.model_management as model_management
import comfy.utils

logger = logging.getLogger(__name__)

class QwenContextProcessor:
    """
    Context image processor for ControlNet-style conditioning with native Qwen models
    
    This processor handles context images separately from vision tokens:
    - edit_image: Uses vision tokens for semantic understanding
    - context_image: Direct latent conditioning for spatial control
    
    This separation enables hybrid workflows where text provides semantic
    direction while context images provide spatial structure.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_images": ("IMAGE", {
                    "tooltip": "Images for spatial conditioning (no vision tokens)"
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE for encoding images to latent space"
                }),
            },
            "optional": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Existing conditioning to augment with context"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0, 
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Context influence strength"
                }),
                "blend_mode": (["replace", "add", "concat"], {
                    "default": "concat",
                    "tooltip": "How to combine with existing conditioning"
                }),
                "resize_mode": (["stretch", "crop", "pad"], {
                    "default": "stretch",
                    "tooltip": "How to handle size mismatches"
                }),
                "debug_context": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show context processing details"
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "LATENT", "DICT")
    RETURN_NAMES = ("conditioning", "context_latents", "debug_info") 
    FUNCTION = "process_context"
    CATEGORY = "QwenImage/Native"
    TITLE = "Qwen Context Processor"
    DESCRIPTION = """
Process context images for ControlNet-style conditioning with native Qwen models.

CONTEXT VS VISION:
- edit_image: Semantic understanding via vision tokens  
- context_image: Spatial control via direct latent conditioning

This enables hybrid workflows combining semantic text guidance
with spatial structure from reference images.
"""

    def _prepare_context_images(self, context_images: torch.Tensor, 
                               vae, resize_mode: str, debug_context: bool) -> torch.Tensor:
        """Prepare context images for VAE encoding"""
        
        if debug_context:
            logger.info(f"Context images shape: {context_images.shape}")
        
        # context_images is [batch, height, width, channels] in 0-1 range
        # We need to process each image in the batch
        batch_size = context_images.shape[0]
        encoded_latents = []
        
        for i in range(batch_size):
            # Get single image [height, width, channels]
            single_image = context_images[i]
            
            # Add batch dimension for VAE: [1, height, width, channels] 
            single_image = single_image.unsqueeze(0)
            
            # Encode to latent space
            with torch.no_grad():
                latent = vae.encode(single_image[:, :, :, :3])  # RGB only
            
            encoded_latents.append(latent)
            
            if debug_context:
                logger.info(f"Image {i}: {single_image.shape} -> {latent.shape}")
        
        # Combine all latents
        if len(encoded_latents) == 1:
            context_latents = encoded_latents[0]
        else:
            # Concatenate along batch dimension
            context_latents = torch.cat(encoded_latents, dim=0)
        
        if debug_context:
            logger.info(f"Final context latents shape: {context_latents.shape}")
        
        return context_latents

    def _combine_conditioning(self, existing_conditioning: Optional[List],
                            context_latents: torch.Tensor,
                            blend_mode: str, strength: float,
                            debug_context: bool) -> List:
        """Combine context latents with existing conditioning"""
        
        # Prepare context conditioning data
        context_data = {
            "context_latents": [context_latents],
            "context_strength": strength,
            "context_blend_mode": blend_mode,
        }
        
        if existing_conditioning is None:
            # Create new conditioning with just context
            if debug_context:
                logger.info("Creating new conditioning with context latents")
            
            # Create minimal text conditioning for context-only workflows
            # This should be replaced with proper text conditioning in practice
            device = model_management.get_torch_device()
            dummy_text_embeds = torch.zeros(1, 77, 3584, device=device)  # Qwen hidden size
            
            from comfy.conds import CONDCrossAttn
            cond = CONDCrossAttn(dummy_text_embeds)
            
            return [[cond, context_data]]
        
        else:
            # Augment existing conditioning
            if debug_context:
                logger.info(f"Augmenting existing conditioning with blend_mode: {blend_mode}")
            
            # Copy existing conditioning and add context data
            augmented_conditioning = []
            
            for cond_entry in existing_conditioning:
                cond, extra_data = cond_entry
                
                # Combine extra data 
                new_extra_data = dict(extra_data) if extra_data else {}
                new_extra_data.update(context_data)
                
                augmented_conditioning.append([cond, new_extra_data])
            
            return augmented_conditioning

    def process_context(
        self,
        context_images: torch.Tensor,
        vae,
        conditioning: Optional[List] = None,
        strength: float = 1.0,
        blend_mode: str = "concat",
        resize_mode: str = "stretch", 
        debug_context: bool = False,
        **kwargs
    ) -> Tuple[List, Dict, Dict]:
        """
        Process context images for ControlNet-style conditioning
        
        This creates spatial conditioning that works alongside text conditioning
        without interfering with vision token processing.
        """
        
        if debug_context:
            logger.info("Processing context images for spatial conditioning")
            logger.info(f"Strength: {strength}, Blend mode: {blend_mode}")
        
        # Prepare context images for VAE encoding
        context_latents = self._prepare_context_images(
            context_images, vae, resize_mode, debug_context
        )
        
        # Combine with existing conditioning
        combined_conditioning = self._combine_conditioning(
            conditioning, context_latents, blend_mode, strength, debug_context
        )
        
        # Create debug info
        debug_info = {
            "context_latents_shape": list(context_latents.shape),
            "context_strength": strength,
            "blend_mode": blend_mode,
            "has_existing_conditioning": conditioning is not None,
            "total_conditioning_entries": len(combined_conditioning)
        }
        
        # Create latent output for direct usage
        latent_output = {"samples": context_latents}
        
        if debug_context:
            logger.info("Context processing completed")
            logger.info(f"Debug info: {debug_info}")
        
        return (combined_conditioning, latent_output, debug_info)

# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenContextProcessor": QwenContextProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenContextProcessor": "Qwen Context Processor"
}