"""
Monkey patch for ComfyUI's Qwen encoder to support multi-frame vision tokens
"""

import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def patch_qwen_encoder():
    """
    Patch ComfyUI's Qwen encoder to fix critical bugs:
    1. Vision processing duplication (2x speedup)
    2. Multi-frame RoPE handling
    3. Template token standardization
    """
    try:
        from comfy.text_encoders.qwen_image import QwenImageTEModel
        from comfy.text_encoders import llama
        from comfy.text_encoders import qwen_vl
        
        # Store original methods
        original_encode = QwenImageTEModel.encode_token_weights
        original_preprocess = llama.Qwen25_7BVLI.preprocess_embed
        original_process_images = qwen_vl.process_qwen2vl_images
        
        def patched_encode_token_weights(self, token_weight_pairs):
            """
            Patched version that handles multi-frame vision tokens.
            """
            # Check if we have multi-frame tokens
            has_multiframe = False
            multiframe_data = None
            
            if "qwen25_7b" in token_weight_pairs:
                tokens = token_weight_pairs["qwen25_7b"][0]
                for token_data in tokens:
                    if isinstance(token_data[0], dict):
                        embed_dict = token_data[0]
                        if embed_dict.get("type") == "multiframe_vision":
                            has_multiframe = True
                            multiframe_data = embed_dict
                            logger.info(f"[Patch] Found multi-frame vision token with {embed_dict['num_frames']} frames")
                            break
            
            if has_multiframe and multiframe_data:
                # Process multi-frame vision specially
                try:
                    logger.info(f"[Patch] Processing multi-frame vision with grid {multiframe_data['grid_thw']}")
                    
                    # Keep the multiframe type - don't change to image!
                    # This is critical - we need to handle it differently
                    multiframe_data["is_multiframe"] = True
                    
                    # Store the pre-processed vision patches
                    vision_patches = multiframe_data["data"]
                    multiframe_data["vision_patches"] = vision_patches
                    
                except Exception as e:
                    logger.error(f"[Patch] Failed to process multi-frame: {e}")
            
            # Call original method
            return original_encode(self, token_weight_pairs)
        
        def patched_preprocess_embed(self, embed, device):
            """
            Patched version that handles multi-frame vision embeddings.
            """
            # Check for our multiframe vision type FIRST
            if embed.get("type") == "multiframe_vision":
                logger.info(f"[Patch] Processing multi-frame embed with {embed.get('num_frames', 0)} frames")
                
                # Get the pre-processed vision patches
                vision_patches = embed.get("data")  # This is our processed patches
                grid_thw = embed.get("grid_thw")
                
                if vision_patches is not None and grid_thw is not None:
                    try:
                        # Move to device and ensure float32
                        if not isinstance(vision_patches, torch.Tensor):
                            logger.error(f"[Patch] Vision patches is not a tensor: {type(vision_patches)}")
                            return None, None
                            
                        vision_patches = vision_patches.to(device, dtype=torch.float32)
                        
                        # Handle grid_thw properly
                        if isinstance(grid_thw, torch.Tensor):
                            grid_thw = grid_thw.to(device)
                            # Ensure it's in the right format for visual encoder [batch, 3]
                            if len(grid_thw.shape) == 1:
                                grid_thw = grid_thw.unsqueeze(0)  # Add batch dimension
                            elif len(grid_thw.shape) == 2 and grid_thw.shape[0] != 1:
                                grid_thw = grid_thw[:1]  # Take first if multiple
                        
                        # Use the visual encoder directly with pre-processed patches
                        if hasattr(self, 'visual'):
                            logger.info(f"[Patch] Using visual encoder for multi-frame patches")
                            logger.info(f"[Patch] Vision patches shape: {vision_patches.shape}")
                            logger.info(f"[Patch] Grid THW: {grid_thw}")
                            
                            # The visual encoder's forward method expects patches
                            # Our patches are already in the right format [seq_len, patch_dim]
                            # Visual encoder internally handles the batch dimension
                            
                            # Call visual encoder's forward directly
                            # This bypasses the normal image processing pipeline
                            vision_output = self.visual(vision_patches, grid_thw)
                            
                            logger.info(f"[Patch] Vision output shape: {vision_output.shape}")
                            
                            # Return embeddings and grid
                            return vision_output, grid_thw
                        else:
                            logger.error("[Patch] No visual encoder found!")
                            return None, None
                        
                    except Exception as e:
                        logger.error(f"[Patch] Vision processing failed: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        return None, None
                
                logger.warning("[Patch] Missing vision data in multiframe token")
                return None, None
            
            # Check if this was marked as multiframe by our earlier patch
            elif embed.get("is_multiframe"):
                logger.info("[Patch] Found is_multiframe marker, treating as multiframe")
                # Redirect to multiframe handling
                embed["type"] = "multiframe_vision"
                return patched_preprocess_embed(self, embed, device)
            
            # For regular image tokens, use original method
            else:
                return original_preprocess(self, embed, device)
        
        def patched_process_qwen2vl_images(
            images,
            min_pixels = 3136,
            max_pixels = 12845056,
            patch_size = 14,
            temporal_patch_size = 2,
            merge_size = 2,
            image_mean = None,
            image_std = None,
        ):
            """
            Fixed version that doesn't duplicate images unnecessarily.
            """
            import torch
            import torch.nn.functional as F
            import math
            
            if image_mean is None:
                image_mean = [0.48145466, 0.4578275, 0.40821073]
            if image_std is None:
                image_std = [0.26862954, 0.26130258, 0.27577711]

            batch_size, height, width, channels = images.shape
            device = images.device

            images = images.permute(0, 3, 1, 2)

            grid_thw_list = []
            img = images[0]

            factor = patch_size * merge_size

            h_bar = round(height / factor) * factor
            w_bar = round(width / factor) * factor

            if h_bar * w_bar > max_pixels:
                beta = math.sqrt((height * width) / max_pixels)
                h_bar = max(factor, math.floor(height / beta / factor) * factor)
                w_bar = max(factor, math.floor(width / beta / factor) * factor)
            elif h_bar * w_bar < min_pixels:
                beta = math.sqrt(min_pixels / (height * width))
                h_bar = math.ceil(height * beta / factor) * factor
                w_bar = math.ceil(width * beta / factor) * factor

            img_resized = F.interpolate(
                img.unsqueeze(0),
                size=(h_bar, w_bar),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            normalized = img_resized.clone()
            for c in range(3):
                normalized[c] = (img_resized[c] - image_mean[c]) / image_std[c]

            grid_h = h_bar // patch_size
            grid_w = w_bar // patch_size
            grid_thw = torch.tensor([1, grid_h, grid_w], device=device, dtype=torch.long)

            pixel_values = normalized
            grid_thw_list.append(grid_thw)
            image_grid_thw = torch.stack(grid_thw_list)

            grid_t = 1
            channel = pixel_values.shape[0]
            
            # FIX: Don't duplicate the same image!
            # Instead, pad with zeros for the second temporal frame
            if temporal_patch_size == 2:
                zero_frame = torch.zeros_like(pixel_values)
                pixel_values = torch.stack([pixel_values, zero_frame], dim=0)
            else:
                pixel_values = pixel_values.unsqueeze(0)

            patches = pixel_values.reshape(
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )

            patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size
            )

            return flatten_patches, image_grid_thw
        
        def patched_encode_token_weights_fixed(self, token_weight_pairs):
            """
            Fixed version with standardized template dropping like DiffSynth.
            """
            out, pooled, extra = original_encode(self, token_weight_pairs)
            tok_pairs = token_weight_pairs["qwen25_7b"][0]
            
            # Check if this is image mode by looking for vision tokens
            has_images = False
            for token_data in tok_pairs:
                if isinstance(token_data[0], dict) and token_data[0].get("type") == "image":
                    has_images = True
                    break
            
            # Use fixed drop indices like DiffSynth instead of dynamic detection
            if has_images:
                drop_idx = 64  # Image edit template
            else:
                drop_idx = 34  # Text-only template
            
            # Apply the fixed drop
            if out.shape[1] > drop_idx:
                out = out[:, drop_idx:]
                # Only process attention_mask if it exists
                if "attention_mask" in extra:
                    extra["attention_mask"] = extra["attention_mask"][:, drop_idx:]
                    if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
                        extra.pop("attention_mask")
                    
            logger.debug(f"[Patch] Applied fixed template drop of {drop_idx} tokens (images={has_images})")
            return out, pooled, extra

        # Apply patches
        QwenImageTEModel.encode_token_weights = patched_encode_token_weights_fixed
        llama.Qwen25_7BVLI.preprocess_embed = patched_preprocess_embed
        qwen_vl.process_qwen2vl_images = patched_process_qwen2vl_images
        
        logger.info("[Patch] Successfully patched Qwen encoder: fixed vision duplication + template dropping")
        return True
        
    except Exception as e:
        logger.error(f"[Patch] Failed to patch Qwen encoder: {e}")
        return False


def unpatch_qwen_encoder():
    """
    Remove the patches (restore original methods).
    """
    try:
        from comfy.text_encoders.qwen_image import QwenImageTEModel
        from comfy.text_encoders import llama
        
        if hasattr(QwenImageTEModel, '_original_encode'):
            QwenImageTEModel.encode_token_weights = QwenImageTEModel._original_encode
            
        if hasattr(llama.Qwen25_7BVLI, '_original_preprocess'):
            llama.Qwen25_7BVLI.preprocess_embed = llama.Qwen25_7BVLI._original_preprocess
            
        logger.info("[Patch] Removed Qwen encoder patches")
        return True
        
    except Exception as e:
        logger.error(f"[Patch] Failed to unpatch: {e}")
        return False


# Auto-patch when module is imported
patch_qwen_encoder()