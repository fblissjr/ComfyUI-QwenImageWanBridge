"""
Unified Qwen2.5-VL Text Encoder
Works with both local models and external APIs
Replaces the old ComfyUI CLIP-based encoder
"""

import logging
import torch
from typing import Optional, Dict, Any, Tuple, Union, List
from .qwen_unified import QwenUnifiedModel

logger = logging.getLogger(__name__)


class QwenUnifiedTextEncoder:
    """
    Unified text encoder that works with QwenUnifiedModel
    Supports both local and API modes
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape",
                    "tooltip": "Your prompt. Templates can be applied via Template Builder."
                }),
                "mode": (["text_to_image", "image_edit"], {
                    "default": "image_edit",
                    "tooltip": "text_to_image: Generate from scratch | image_edit: Modify existing image"
                }),
            },
            "optional": {
                "qwen_model": ("QWEN_MODEL", {
                    "tooltip": "Connect from Qwen2.5-VL Unified Loader (preferred)"
                }),
                "clip": ("CLIP", {
                    "tooltip": "Connect from QwenVLCLIPLoader (backward compatibility)"
                }),
                "edit_image": ("IMAGE", {
                    "tooltip": "Image to edit/reference. Can be from Multi-Reference Handler."
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE for encoding reference images"
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show detailed processing info in console"
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "QwenImage/Encoding"
    TITLE = "Qwen2.5-VL Unified Text Encoder"
    DESCRIPTION = """
Unified text encoder for Qwen2.5-VL that works with both local models and APIs.

For image editing:
- Connect edit_image for vision-based editing
- High denoise (0.9-1.0) = semantic editing using vision understanding
- Low denoise (0.3-0.7) = structure preservation
"""
    
    def encode(self, text: str, mode: str = "image_edit",
              qwen_model: QwenUnifiedModel = None, clip=None,
              edit_image: Optional[torch.Tensor] = None, vae=None,
              debug_mode: bool = False) -> Tuple[Any]:
        
        images_for_processing = []
        ref_latent = None
        original_text = text
        
        # Determine which model to use
        model_to_use = None
        model_type = "unknown"
        
        if qwen_model is not None:
            model_to_use = qwen_model
            model_type = 'local' if qwen_model.is_local else 'api'
        elif clip is not None:
            model_to_use = clip
            model_type = 'clip'
        else:
            raise ValueError("Must connect either qwen_model (QwenUnifiedLoader) or clip (QwenVLCLIPLoader)")
        
        if debug_mode:
            logger.info(f"[Encoder] Mode: {mode}, Model type: {model_type}")
            logger.info(f"[Encoder] Text input: '{text[:100]}...'")
        
        # Prepare images for processing
        if mode == "image_edit" and edit_image is not None:
            if debug_mode:
                logger.info(f"[Encoder] Processing edit image: {edit_image.shape}")
            
            images_for_processing = [edit_image]
            
            # Encode reference latent if VAE provided
            if vae is not None:
                ref_latent = vae.encode(edit_image[:, :, :, :3])
                if debug_mode:
                    logger.info(f"[Encoder] Encoded reference latent: {ref_latent.shape}")
        
        # Text is now expected to come pre-formatted from Template Builder or raw user input
        # No automatic template application - keeps the architecture clean
        
        if debug_mode:
            logger.info(f"[Encoder] Final text for processing: '{text[:100]}...'")
        
        # Handle different model types
        if model_type == 'clip':
            conditioning = self._encode_clip(clip, text, images_for_processing, debug_mode)
        elif qwen_model.is_local:
            conditioning = self._encode_local(qwen_model, text, images_for_processing, debug_mode)
        else:
            conditioning = self._encode_api(qwen_model, text, images_for_processing, debug_mode)
        
        # Add reference latents to conditioning if available
        if ref_latent is not None:
            # For now, return conditioning as-is
            # TODO: Integrate reference latents into conditioning properly
            if debug_mode:
                logger.info("[Encoder] Reference latent available but not yet integrated")
        
        if debug_mode:
            if isinstance(conditioning, list) and len(conditioning) > 0:
                logger.info(f"[Encoder] Generated conditioning: ComfyUI format with tensor shape {conditioning[0][0].shape}")
            else:
                logger.info(f"[Encoder] Generated conditioning shape: {conditioning.shape if hasattr(conditioning, 'shape') else type(conditioning)}")
        
        return (conditioning,)
    
    def _encode_local(self, qwen_model: QwenUnifiedModel, text: str, 
                     images: List, debug_mode: bool) -> torch.Tensor:
        """Encode using local model"""
        
        if debug_mode:
            logger.info("[Encoder] Using local model for encoding")
        
        # Tokenize text and images
        tokens = qwen_model.tokenize(text, images)
        
        # Encode to conditioning
        conditioning = qwen_model.encode_from_tokens(tokens)
        
        if debug_mode:
            if isinstance(conditioning, list) and len(conditioning) > 0:
                logger.info(f"[Encoder] Local encoding complete: ComfyUI format with tensor shape {conditioning[0][0].shape}")
            else:
                logger.info(f"[Encoder] Local encoding complete: {conditioning.shape if hasattr(conditioning, 'shape') else type(conditioning)}")
        
        return conditioning
    
    def _encode_clip(self, clip, text: str, images: List, debug_mode: bool) -> torch.Tensor:
        """Encode using CLIP model (backward compatibility)"""
        
        if debug_mode:
            logger.info("[Encoder] Using CLIP model for encoding")
        
        # Use standard CLIP tokenization and encoding
        try:
            # Convert images for CLIP tokenization
            images_for_clip = []
            if images:
                images_for_clip = images  # CLIP expects the raw image tensors
                
            # Tokenize using CLIP
            tokens = clip.tokenize(text, images=images_for_clip)
            
            # Encode using CLIP
            conditioning = clip.encode_from_tokens_scheduled(tokens)
            
            if debug_mode:
                logger.info(f"[Encoder] CLIP encoding complete: {conditioning.shape if hasattr(conditioning, 'shape') else type(conditioning)}")
            
            return conditioning
            
        except Exception as e:
            logger.error(f"CLIP encoding failed: {e}")
            # Return placeholder conditioning
            return torch.zeros(1, 1, 768, dtype=torch.float32)  # Standard CLIP size
    
    def _encode_api(self, qwen_model: QwenUnifiedModel, text: str, 
                   images: List, debug_mode: bool) -> torch.Tensor:
        """Encode using API model"""
        
        if debug_mode:
            logger.info("[Encoder] Using API model for encoding")
        
        # For API mode, we create a conditioning tensor that contains the text and images
        # This will be processed during the diffusion sampling step
        
        # Create a structured conditioning object
        conditioning_data = {
            "text": text,
            "images": images,
            "model_info": {
                "endpoint": qwen_model.api_endpoint,
                "api_key": qwen_model.api_key
            },
            "type": "qwen_api_conditioning"
        }
        
        # Create a placeholder tensor (will be replaced during sampling)
        # Include conditioning data as tensor attribute
        conditioning_tensor = torch.zeros(1, 1, 3584, dtype=torch.float32)
        conditioning_tensor._qwen_api_data = conditioning_data
        
        if debug_mode:
            logger.info("[Encoder] API conditioning prepared")
        
        return conditioning_tensor


# Removed QwenUnifiedSampler - it was redundant
# Users should use standard KSampler with our conditioning


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenUnifiedTextEncoder": QwenUnifiedTextEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenUnifiedTextEncoder": "Qwen2.5-VL Unified Text Encoder",
}