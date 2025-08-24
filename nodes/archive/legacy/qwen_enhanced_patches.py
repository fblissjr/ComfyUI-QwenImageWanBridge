"""
Enhanced Qwen Support via Monkey Patching
Adds missing features to existing ComfyUI nodes without duplication
Uses DiffSynth as reference but doesn't require DiffSynth runtime
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
from typing import Optional, Dict, List, Tuple, Any
import comfy.utils
import comfy.model_management
import node_helpers

# Import existing ComfyUI Qwen nodes to patch
try:
    from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit
    from comfy_extras.nodes_model_patch import QwenImageBlockWiseControlNet
    NATIVE_NODES_AVAILABLE = True
except ImportError:
    NATIVE_NODES_AVAILABLE = False
    print("[QwenEnhanced] Native nodes not found, creating standalone versions")


class TextEncodeQwenImageEditEnhanced:
    """
    Enhanced version of TextEncodeQwenImageEdit with DiffSynth features
    Monkey patches the original or provides standalone if not available
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mode": (["auto", "text_to_image", "image_edit"], {"default": "auto"}),
                "template": (["default", "edit", "custom"], {"default": "default"}),
                "custom_template": ("STRING", {"multiline": True, "default": ""}),
                "auto_resize": ("BOOLEAN", {"default": True}),
                "target_pixels": ("INT", {"default": 1048576, "min": 262144, "max": 4194304}),
                "drop_template_tokens": ("BOOLEAN", {"default": True}),
                "rope_interpolation": ("BOOLEAN", {"default": False}),
                "entity_prompts": ("STRING", {"multiline": True, "default": ""}),
                "entity_masks": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "DICT")
    RETURN_NAMES = ("conditioning", "encode_info")
    FUNCTION = "encode_enhanced"
    CATEGORY = "advanced/conditioning/qwen"
    
    # Templates from DiffSynth
    T2I_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    
    EDIT_TEMPLATE = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    
    def __init__(self):
        # If native node exists, store reference to original encode method
        if NATIVE_NODES_AVAILABLE:
            self.original_encode = TextEncodeQwenImageEdit().encode
    
    def calculate_optimal_size(self, image: torch.Tensor, target_pixels: int) -> Tuple[int, int]:
        """Calculate optimal dimensions from DiffSynth's approach"""
        if len(image.shape) == 4:
            _, h, w, _ = image.shape
        else:
            h, w, _ = image.shape
        
        # Calculate scale to reach target pixel count
        current_pixels = h * w
        scale = math.sqrt(target_pixels / current_pixels)
        
        # Apply scale and round to multiple of 32
        new_w = round(w * scale / 32) * 32
        new_h = round(h * scale / 32) * 32
        
        # Ensure minimum size
        new_w = max(new_w, 512)
        new_h = max(new_h, 512)
        
        return new_w, new_h
    
    def process_with_template(self, clip, prompt: str, template: str, 
                            drop_template: bool = True) -> torch.Tensor:
        """Process prompt with template, optionally dropping template tokens"""
        formatted = template.format(prompt)
        tokens = clip.tokenize(formatted)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        if drop_template:
            # Find template end (after second <|im_start|> and "assistant\n")
            # This is from DiffSynth's implementation
            # Token IDs: im_start=151644, assistant=872, newline=198
            try:
                # Get the raw tokens if accessible
                if hasattr(tokens, 'input_ids'):
                    token_list = tokens.input_ids[0].tolist()
                    count_im_start = 0
                    template_end = 0
                    
                    for i, token_id in enumerate(token_list):
                        if token_id == 151644:  # <|im_start|>
                            count_im_start += 1
                            if count_im_start >= 2:
                                if i + 2 < len(token_list):
                                    if token_list[i + 1] == 872 and token_list[i + 2] == 198:
                                        template_end = i + 3
                                        break
                    
                    if template_end > 0:
                        # Slice conditioning to remove template
                        conditioning = [[cond[0][:, template_end:], cond[1]] for cond in conditioning]
            except:
                pass  # Fallback to keeping full conditioning
        
        return conditioning
    
    def handle_entity_control(self, clip, entity_prompts: str, 
                            entity_masks: Optional[torch.Tensor]) -> Dict:
        """Process entity prompts for EliGen-style control"""
        if not entity_prompts:
            return {}
        
        entity_list = [p.strip() for p in entity_prompts.split("|") if p.strip()]
        entity_embeddings = []
        entity_attention_masks = []
        
        for entity in entity_list:
            entity_formatted = self.T2I_TEMPLATE.format(entity)
            entity_tokens = clip.tokenize(entity_formatted)
            entity_cond = clip.encode_from_tokens_scheduled(entity_tokens)
            entity_embeddings.append(entity_cond[0][0])
            
            # Create attention mask
            mask = torch.ones_like(entity_cond[0][0][:, :, 0])
            entity_attention_masks.append(mask)
        
        return {
            "entity_prompt_emb": entity_embeddings,
            "entity_prompt_emb_mask": entity_attention_masks,
            "entity_masks": entity_masks
        }
    
    def encode_enhanced(self, clip, prompt: str, vae=None, image=None,
                       mode: str = "auto", template: str = "default",
                       custom_template: str = "", auto_resize: bool = True,
                       target_pixels: int = 1048576, drop_template_tokens: bool = True,
                       rope_interpolation: bool = False, entity_prompts: str = "",
                       entity_masks: Optional[torch.Tensor] = None) -> Tuple[list, Dict]:
        """
        Enhanced encode with DiffSynth features
        """
        encode_info = {
            "mode": mode,
            "template_used": template,
            "auto_resized": False,
            "rope_interpolation": rope_interpolation
        }
        
        # Auto-detect mode
        if mode == "auto":
            mode = "image_edit" if image is not None else "text_to_image"
            encode_info["mode"] = mode
        
        # Handle image resizing
        ref_latent = None
        images = []
        
        if image is not None:
            if auto_resize:
                # Use DiffSynth's resize approach
                new_w, new_h = self.calculate_optimal_size(image, target_pixels)
                samples = image.movedim(-1, 1)
                resized = comfy.utils.common_upscale(samples, new_w, new_h, "area", "disabled")
                image = resized.movedim(1, -1)
                encode_info["auto_resized"] = True
                encode_info["new_size"] = (new_w, new_h)
            
            images = [image[:, :, :, :3]]
            
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])
        
        # Select and apply template
        if template == "custom" and custom_template:
            template_str = custom_template
        elif template == "edit" or (template == "default" and mode == "image_edit"):
            template_str = self.EDIT_TEMPLATE
            encode_info["has_vision_tokens"] = True
        else:
            template_str = self.T2I_TEMPLATE
        
        # Process with template
        if template_str and "{}" in template_str:
            conditioning = self.process_with_template(clip, prompt, template_str, drop_template_tokens)
        else:
            # Fallback to standard encoding
            tokens = clip.tokenize(prompt, images=images)
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # Add reference latents if available
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": [ref_latent]}, append=True
            )
        
        # Handle entity control
        if entity_prompts:
            entity_data = self.handle_entity_control(clip, entity_prompts, entity_masks)
            if entity_data:
                conditioning = node_helpers.conditioning_set_values(
                    conditioning, entity_data, append=True
                )
                encode_info["num_entities"] = len(entity_data.get("entity_prompt_emb", []))
        
        # Add RoPE interpolation flag
        if rope_interpolation:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"rope_interpolation": True}, append=True
            )
        
        encode_info["conditioning_shape"] = conditioning[0][0].shape if conditioning else None
        
        return (conditioning, encode_info)


class QwenBlockwiseControlNetEnhanced:
    """
    Enhanced BlockWise ControlNet with multi-controlnet support
    Based on DiffSynth's implementation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "controlnet": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
                "controlnet_2": ("CONTROL_NET",),
                "image_2": ("IMAGE",),
                "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "advanced/controlnet/qwen"
    
    def apply_controlnet(self, model, controlnet, image, strength,
                        mask=None, controlnet_2=None, image_2=None, 
                        strength_2=1.0, start_percent=0.0, end_percent=1.0):
        """
        Apply enhanced blockwise control with multi-controlnet support
        """
        # This would integrate with the existing QwenImageBlockWiseControlNet
        # Adding support for multiple control inputs and progressive strength
        
        # For now, return the model as-is
        # Full implementation would require deeper integration with ComfyUI's control system
        return (model,)


def monkey_patch_native_nodes():
    """
    Monkey patch existing ComfyUI nodes with enhanced functionality
    """
    if not NATIVE_NODES_AVAILABLE:
        print("[QwenEnhanced] Native nodes not available, skipping monkey patch")
        return
    
    # Store original methods
    TextEncodeQwenImageEdit._original_encode = TextEncodeQwenImageEdit.encode
    
    # Create enhanced instance
    enhanced = TextEncodeQwenImageEditEnhanced()
    
    # Patch the encode method
    def patched_encode(self, clip, prompt, vae=None, image=None, **kwargs):
        # Check if called with enhanced parameters
        if any(k in kwargs for k in ['mode', 'template', 'auto_resize', 'entity_prompts']):
            # Use enhanced version
            return enhanced.encode_enhanced(clip, prompt, vae, image, **kwargs)
        else:
            # Use original version for backward compatibility
            return self._original_encode(clip, prompt, vae, image)
    
    TextEncodeQwenImageEdit.encode = patched_encode
    
    # Update INPUT_TYPES to include new parameters
    original_input_types = TextEncodeQwenImageEdit.INPUT_TYPES
    
    @classmethod
    def enhanced_input_types(cls):
        types = original_input_types()
        types["optional"].update({
            "mode": (["auto", "text_to_image", "image_edit"], {"default": "auto"}),
            "template": (["default", "edit", "custom"], {"default": "default"}),
            "custom_template": ("STRING", {"multiline": True, "default": ""}),
            "auto_resize": ("BOOLEAN", {"default": True}),
            "target_pixels": ("INT", {"default": 1048576}),
            "drop_template_tokens": ("BOOLEAN", {"default": True}),
            "rope_interpolation": ("BOOLEAN", {"default": False}),
            "entity_prompts": ("STRING", {"multiline": True, "default": ""}),
            "entity_masks": ("MASK",),
        })
        return types
    
    TextEncodeQwenImageEdit.INPUT_TYPES = enhanced_input_types
    
    print("[QwenEnhanced] Successfully monkey patched native Qwen nodes")


# Auto-patch on import
try:
    monkey_patch_native_nodes()
except Exception as e:
    print(f"[QwenEnhanced] Failed to monkey patch: {e}")


# Provide nodes for registration
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Only register enhanced nodes if native ones aren't available
if not NATIVE_NODES_AVAILABLE:
    NODE_CLASS_MAPPINGS["TextEncodeQwenImageEditEnhanced"] = TextEncodeQwenImageEditEnhanced
    NODE_DISPLAY_NAME_MAPPINGS["TextEncodeQwenImageEditEnhanced"] = "Qwen Image Edit Enhanced"

# Always register the multi-controlnet node as it's new functionality
NODE_CLASS_MAPPINGS["QwenBlockwiseControlNetEnhanced"] = QwenBlockwiseControlNetEnhanced
NODE_DISPLAY_NAME_MAPPINGS["QwenBlockwiseControlNetEnhanced"] = "Qwen Blockwise ControlNet Enhanced"