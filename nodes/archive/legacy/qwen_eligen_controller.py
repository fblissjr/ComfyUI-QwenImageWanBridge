"""
Qwen-Image EliGen Controller
Entity-level generation control for precise regional generation
Supports EliGen V2 with improved entity separation
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import folder_paths
import comfy.model_management

class QwenEliGenController:
    """
    Entity-level generation controller for Qwen-Image
    Allows precise control over multiple entities with masks
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "global_prompt": ("STRING", {"multiline": True, "default": "A beautiful scene"}),
                "enable_on_negative": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "entity_1_prompt": ("STRING", {"multiline": False, "default": ""}),
                "entity_1_mask": ("MASK",),
                "entity_2_prompt": ("STRING", {"multiline": False, "default": ""}),
                "entity_2_mask": ("MASK",),
                "entity_3_prompt": ("STRING", {"multiline": False, "default": ""}),
                "entity_3_mask": ("MASK",),
                "entity_4_prompt": ("STRING", {"multiline": False, "default": ""}),
                "entity_4_mask": ("MASK",),
                "lora_path": ("STRING", {"default": "models/DiffSynth-Studio/Qwen-Image-EliGen-V2/model.safetensors"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "DICT")
    RETURN_NAMES = ("model", "positive", "negative", "eligen_info")
    FUNCTION = "apply_eligen"
    CATEGORY = "QwenImage/EliGen"
    
    def prepare_entity_masks(self, masks: List[torch.Tensor], height: int, width: int) -> torch.Tensor:
        """Prepare and stack entity masks"""
        processed_masks = []
        
        for mask in masks:
            if mask is None:
                continue
                
            # Ensure mask is 2D
            if len(mask.shape) == 3:
                mask = mask.squeeze(0)
            elif len(mask.shape) == 4:
                mask = mask[0, 0]
            
            # Resize to latent space (divide by 8 for VAE)
            latent_h, latent_w = height // 8, width // 8
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(latent_h, latent_w),
                mode='nearest'
            ).squeeze()
            
            # Binarize mask
            mask = (mask > 0.5).float()
            processed_masks.append(mask)
        
        if not processed_masks:
            return None
        
        # Stack masks (B, N, 1, H, W)
        masks_tensor = torch.stack(processed_masks).unsqueeze(0).unsqueeze(2)
        return masks_tensor
    
    def encode_entity_prompts(self, clip, entity_prompts: List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode each entity prompt separately"""
        embeddings = []
        attention_masks = []
        
        for prompt in entity_prompts:
            if not prompt:
                continue
            
            # Encode with Qwen template
            template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            formatted = template.format(prompt)
            
            tokens = clip.tokenize(formatted)
            cond = clip.encode(tokens)
            
            # Extract embeddings and create attention mask
            if isinstance(cond, tuple):
                emb, pooled = cond
            else:
                emb = cond
            
            embeddings.append(emb)
            
            # Create attention mask (all ones for valid tokens)
            mask = torch.ones(emb.shape[0], emb.shape[1], device=emb.device)
            attention_masks.append(mask)
        
        return embeddings, attention_masks
    
    def create_attention_mask(self, entity_masks: torch.Tensor, entity_embeddings: List[torch.Tensor],
                            global_embedding: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Create attention mask for entity-aware generation"""
        device = global_embedding.device
        batch_size = 1
        
        # Calculate sequence lengths
        entity_seq_lens = [emb.shape[1] for emb in entity_embeddings]
        global_seq_len = global_embedding.shape[1]
        total_seq_len = sum(entity_seq_lens) + global_seq_len
        
        # Image sequence length
        image_seq_len = (height // 16) * (width // 16)
        total_len = total_seq_len + image_seq_len
        
        # Initialize attention mask
        attention_mask = torch.ones((batch_size, total_len, total_len), dtype=torch.bool, device=device)
        
        # Set up prompt-to-prompt masking (entities don't attend to each other)
        cumsum = [0]
        for length in entity_seq_lens:
            cumsum.append(cumsum[-1] + length)
        cumsum.append(cumsum[-1] + global_seq_len)
        
        for i in range(len(entity_seq_lens) + 1):
            for j in range(len(entity_seq_lens) + 1):
                if i != j:
                    start_i, end_i = cumsum[i], cumsum[i + 1]
                    start_j, end_j = cumsum[j], cumsum[j + 1]
                    attention_mask[:, start_i:end_i, start_j:end_j] = False
        
        # Set up entity-to-image masking based on masks
        if entity_masks is not None:
            image_start = total_seq_len
            
            for i, mask in enumerate(entity_masks[0]):
                if i >= len(entity_seq_lens):
                    break
                    
                # Reshape mask to image patches
                mask_patches = mask.view(-1) > 0.5
                
                prompt_start = cumsum[i]
                prompt_end = cumsum[i + 1]
                
                # Entity prompt attends only to its masked region
                for j, is_masked in enumerate(mask_patches):
                    if not is_masked:
                        attention_mask[:, prompt_start:prompt_end, image_start + j] = False
                        attention_mask[:, image_start + j, prompt_start:prompt_end] = False
        
        # Convert to proper format
        attention_mask = attention_mask.float()
        attention_mask[attention_mask == 0] = float('-inf')
        attention_mask[attention_mask == 1] = 0
        
        return attention_mask.unsqueeze(1)
    
    def apply_eligen(self, model, clip, global_prompt: str, enable_on_negative: bool = False,
                    entity_1_prompt: str = "", entity_1_mask: Optional[torch.Tensor] = None,
                    entity_2_prompt: str = "", entity_2_mask: Optional[torch.Tensor] = None,
                    entity_3_prompt: str = "", entity_3_mask: Optional[torch.Tensor] = None,
                    entity_4_prompt: str = "", entity_4_mask: Optional[torch.Tensor] = None,
                    lora_path: str = "", lora_strength: float = 1.0) -> Tuple:
        """
        Apply EliGen entity control to the model
        """
        device = comfy.model_management.get_torch_device()
        
        # Collect entity prompts and masks
        entity_prompts = []
        entity_masks = []
        
        for prompt, mask in [(entity_1_prompt, entity_1_mask),
                            (entity_2_prompt, entity_2_mask),
                            (entity_3_prompt, entity_3_mask),
                            (entity_4_prompt, entity_4_mask)]:
            if prompt and mask is not None:
                entity_prompts.append(prompt)
                entity_masks.append(mask)
        
        eligen_info = {
            "num_entities": len(entity_prompts),
            "entity_prompts": entity_prompts,
            "global_prompt": global_prompt,
            "lora_loaded": False
        }
        
        # Load LoRA if specified
        if lora_path and len(entity_prompts) > 0:
            try:
                import comfy.sd
                model = comfy.sd.load_lora_for_models(
                    model, clip, lora_path, lora_strength, lora_strength
                )[0]
                eligen_info["lora_loaded"] = True
                eligen_info["lora_strength"] = lora_strength
            except Exception as e:
                print(f"[QwenEliGen] Failed to load LoRA: {e}")
        
        # Encode global prompt
        global_tokens = clip.tokenize(global_prompt)
        global_cond = clip.encode(global_tokens)
        
        if isinstance(global_cond, tuple):
            global_embedding, global_pooled = global_cond
        else:
            global_embedding = global_cond
            global_pooled = None
        
        # If no entities, return standard conditioning
        if not entity_prompts:
            positive = [[global_embedding, {}]]
            negative = [[torch.zeros_like(global_embedding), {}]]
            return (model, positive, negative, eligen_info)
        
        # Encode entity prompts
        entity_embeddings, entity_attention_masks = self.encode_entity_prompts(clip, entity_prompts)
        
        # Prepare entity masks (assume 1024x1024 default, will be overridden in sampler)
        height, width = 1024, 1024
        if entity_masks:
            # Get dimensions from first mask
            if len(entity_masks[0].shape) >= 2:
                height, width = entity_masks[0].shape[-2:]
        
        entity_masks_tensor = self.prepare_entity_masks(entity_masks, height, width)
        
        # Combine embeddings
        all_embeddings = entity_embeddings + [global_embedding]
        combined_embedding = torch.cat(all_embeddings, dim=1)
        
        # Create attention mask
        attention_mask = self.create_attention_mask(
            entity_masks_tensor, entity_embeddings, global_embedding, height, width
        )
        
        # Create conditioning with entity information
        positive_dict = {
            "entity_prompt_emb": entity_embeddings,
            "entity_prompt_emb_mask": entity_attention_masks,
            "entity_masks": entity_masks_tensor,
            "attention_mask": attention_mask
        }
        
        positive = [[combined_embedding, positive_dict]]
        
        # Handle negative conditioning
        if enable_on_negative:
            # Apply entity control to negative as well
            negative = [[combined_embedding, positive_dict]]
        else:
            # Standard negative
            negative_tokens = clip.tokenize("")
            negative_cond = clip.encode(negative_tokens)
            if isinstance(negative_cond, tuple):
                negative_embedding = negative_cond[0]
            else:
                negative_embedding = negative_cond
            negative = [[negative_embedding, {}]]
        
        eligen_info["entity_masks_shape"] = list(entity_masks_tensor.shape) if entity_masks_tensor is not None else None
        eligen_info["combined_embedding_shape"] = list(combined_embedding.shape)
        
        return (model, positive, negative, eligen_info)


class QwenEliGenMaskCreator:
    """
    Helper node to create masks for EliGen control
    Supports drawing simple shapes or converting from images
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "shape": (["rectangle", "circle", "polygon", "from_image"], {"default": "rectangle"}),
            },
            "optional": {
                "x": ("INT", {"default": 256, "min": 0, "max": 4096}),
                "y": ("INT", {"default": 256, "min": 0, "max": 4096}),
                "size_w": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "size_h": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "input_mask": ("MASK",),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "create_mask"
    CATEGORY = "QwenImage/EliGen"
    
    def create_mask(self, width: int, height: int, shape: str,
                   x: int = 256, y: int = 256, size_w: int = 512, size_h: int = 512,
                   input_mask: Optional[torch.Tensor] = None,
                   feather: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create or process masks for entity control
        """
        if shape == "from_image" and input_mask is not None:
            # Use provided mask
            mask = input_mask
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        else:
            # Create new mask
            mask = torch.zeros((1, height, width), dtype=torch.float32)
            
            if shape == "rectangle":
                # Draw rectangle
                x_end = min(x + size_w, width)
                y_end = min(y + size_h, height)
                mask[0, y:y_end, x:x_end] = 1.0
                
            elif shape == "circle":
                # Draw circle
                center_x = x + size_w // 2
                center_y = y + size_h // 2
                radius = min(size_w, size_h) // 2
                
                Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
                dist = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
                mask[0] = (dist <= radius).float()
                
            elif shape == "polygon":
                # Simple diamond/rhombus shape as example polygon
                center_x = x + size_w // 2
                center_y = y + size_h // 2
                
                for i in range(height):
                    for j in range(width):
                        # Diamond shape equation
                        if (abs(i - center_y) / (size_h / 2) + 
                            abs(j - center_x) / (size_w / 2)) <= 1:
                            mask[0, i, j] = 1.0
        
        # Apply feathering if requested
        if feather > 0:
            import scipy.ndimage
            mask_np = mask[0].cpu().numpy()
            mask_np = scipy.ndimage.gaussian_filter(mask_np, sigma=feather)
            mask[0] = torch.from_numpy(mask_np)
        
        # Create preview image
        preview = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        
        return (mask, preview)


NODE_CLASS_MAPPINGS = {
    "QwenEliGenController": QwenEliGenController,
    "QwenEliGenMaskCreator": QwenEliGenMaskCreator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenEliGenController": "Qwen EliGen Controller",
    "QwenEliGenMaskCreator": "Qwen EliGen Mask Creator",
}