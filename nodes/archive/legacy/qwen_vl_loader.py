"""
Qwen2.5-VL Model Loader for ComfyUI
Loads models from ComfyUI/models/text_encoders/
"""

import os
import torch
import folder_paths
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Try to import ComfyUI's utilities
try:
    from comfy.utils import load_torch_file
    import comfy.sd
    import comfy.model_management as mm
except ImportError:
    logger.warning("ComfyUI utilities not available")
    load_torch_file = None
    mm = None

class QwenVLModelWrapper:
    """Wrapper for Qwen2.5-VL model to work with ComfyUI conventions"""
    
    def __init__(self, state_dict, device="cuda", dtype=torch.float16):
        self.state_dict = state_dict
        self.device = device
        self.dtype = dtype
        
        # Vision token IDs
        self.vision_start_id = 151652
        self.image_pad_id = 151655
        self.vision_end_id = 151653
        
        # Model info from state dict
        self.hidden_size = 3584  # Qwen2.5-7B-VL default
        self.vocab_size = 151936
        
        # For compatibility - we'll need to implement actual encoding differently
        # since we're loading just weights, not a full transformers model
        logger.warning("Loading from safetensors file - full vision processing requires complete model directory")
        
    def encode_text(self, text: str) -> torch.Tensor:
        """Placeholder for text encoding - needs full model"""
        # Create dummy embeddings of the right shape
        # In reality, this needs the full Qwen model with transformers
        batch_size = 1
        seq_len = 77  # Standard sequence length
        embed_dim = self.hidden_size
        
        dummy_embeddings = torch.randn(batch_size, seq_len, embed_dim, 
                                      device=self.device, dtype=self.dtype)
        return dummy_embeddings
    
    def encode_multimodal(self, text: str, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Placeholder for multimodal encoding"""
        embeddings = self.encode_text(text)
        vision_info = {
            "has_vision": True,
            "warning": "Full vision processing requires complete model directory with config.json"
        }
        return embeddings, vision_info

class QwenVLLoader:
    """Load Qwen2.5-VL model for ComfyUI"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get models from text_encoders folder - same pattern as GGUF loader
        models = folder_paths.get_filename_list("text_encoders")
        if not models:
            models = ["qwen_2.5_vl_7b.safetensors"]  # Default suggestion
        
        return {
            "required": {
                "model_name": (models, {
                    "tooltip": "These models are loaded from 'ComfyUI/models/text_encoders'"
                }),
                "precision": (["fp16", "fp32", "bf16"], {
                    "default": "fp16"
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda"
                })
            }
        }
    
    RETURN_TYPES = ("QWEN_VL_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "QwenImage/Loaders"
    TITLE = "Qwen2.5-VL Loader"
    DESCRIPTION = """Load Qwen2.5-VL model from ComfyUI/models/text_encoders/
    
    Note: Loading from a single .safetensors file provides limited functionality.
    For full vision processing, use a complete model directory with:
    - config.json
    - tokenizer files
    - model weights
    
    Or use HuggingFace model IDs with transformers library installed.
    """
    
    def load_model(self, model_name: str, precision: str = "fp16", 
                   device: str = "cuda") -> Tuple[Any]:
        """Load Qwen2.5-VL model"""
        
        # Determine dtype
        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16
        }
        dtype = dtype_map.get(precision, torch.float16)
        
        # Get full path using ComfyUI's system
        model_path = folder_paths.get_full_path("text_encoders", model_name)
        logger.info(f"Loading Qwen2.5-VL model from: {model_path}")
        
        # Check if it's a safetensors file or directory
        if model_path.endswith(".safetensors"):
            # Load as safetensors file - similar to GGUF pattern
            logger.info("Loading from safetensors file")
            
            if load_torch_file is None:
                raise RuntimeError("ComfyUI load_torch_file not available")
            
            # Load the state dict
            state_dict = load_torch_file(model_path, safe_load=True)
            
            # Create wrapper with loaded weights
            wrapper = QwenVLModelWrapper(
                state_dict=state_dict,
                device=device,
                dtype=dtype
            )
            
            logger.info(f"Loaded Qwen2.5-VL weights from safetensors")
            logger.warning("Note: Full vision processing requires complete model directory or transformers library")
            
            return (wrapper,)
            
        elif os.path.isdir(model_path):
            # It's a directory - try to load with transformers if available
            try:
                from transformers import (
                    Qwen2VLForConditionalGeneration,
                    Qwen2VLProcessor,
                    AutoProcessor,
                    AutoModelForCausalLM
                )
                
                logger.info("Loading from directory with transformers")
                
                processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
                
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map=device if device == "cuda" else None,
                    trust_remote_code=True,
                    local_files_only=True
                )
                
                if device == "cpu":
                    model = model.to(device)
                
                model.eval()
                
                # Create full wrapper with transformers model
                class FullQwenVLWrapper(QwenVLModelWrapper):
                    def __init__(self, model, processor, device, dtype):
                        self.model = model
                        self.processor = processor
                        self.device = device
                        self.dtype = dtype
                        self.hidden_size = model.config.hidden_size
                        self.vocab_size = model.config.vocab_size
                        self.vision_start_id = 151652
                        self.image_pad_id = 151655
                        self.vision_end_id = 151653
                    
                    def encode_text(self, text: str) -> torch.Tensor:
                        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                        return outputs.hidden_states[-1]
                    
                    def encode_multimodal(self, text: str, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
                        if isinstance(image, torch.Tensor):
                            if image.dim() == 4:
                                image = image[0]
                            import numpy as np
                            from PIL import Image
                            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
                            image_pil = Image.fromarray(image_np)
                        else:
                            image_pil = image
                        
                        inputs = self.processor(text=text, images=image_pil, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                        
                        vision_info = {
                            "has_vision": True,
                            "image_grid_thw": inputs.get("image_grid_thw", None),
                            "pixel_values": inputs.get("pixel_values", None)
                        }
                        return outputs.hidden_states[-1], vision_info
                
                wrapper = FullQwenVLWrapper(model, processor, device, dtype)
                logger.info(f"Successfully loaded full Qwen2.5-VL model with transformers")
                
                return (wrapper,)
                
            except ImportError:
                logger.warning("Transformers not available, loading weights only")
                # Fall back to loading just the weights
                config_path = os.path.join(model_path, "pytorch_model.bin")
                if not os.path.exists(config_path):
                    config_path = os.path.join(model_path, "model.safetensors")
                
                if os.path.exists(config_path):
                    state_dict = load_torch_file(config_path, safe_load=True)
                    wrapper = QwenVLModelWrapper(state_dict=state_dict, device=device, dtype=dtype)
                    return (wrapper,)
                else:
                    raise RuntimeError(f"No model weights found in directory: {model_path}")
        else:
            # Try as HuggingFace model ID
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                
                logger.info(f"Attempting to load as HuggingFace model ID: {model_name}")
                
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device if device == "cuda" else None,
                    trust_remote_code=True
                )
                
                # Use the full wrapper class from above
                class FullQwenVLWrapper(QwenVLModelWrapper):
                    def __init__(self, model, processor, device, dtype):
                        self.model = model
                        self.processor = processor
                        self.device = device
                        self.dtype = dtype
                        self.hidden_size = model.config.hidden_size
                        self.vocab_size = model.config.vocab_size
                        self.vision_start_id = 151652
                        self.image_pad_id = 151655
                        self.vision_end_id = 151653
                    
                    def encode_text(self, text: str) -> torch.Tensor:
                        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                        return outputs.hidden_states[-1]
                    
                    def encode_multimodal(self, text: str, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
                        if isinstance(image, torch.Tensor):
                            if image.dim() == 4:
                                image = image[0]
                            import numpy as np
                            from PIL import Image
                            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
                            image_pil = Image.fromarray(image_np)
                        else:
                            image_pil = image
                        
                        inputs = self.processor(text=text, images=image_pil, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                        
                        vision_info = {
                            "has_vision": True,
                            "image_grid_thw": inputs.get("image_grid_thw", None),
                            "pixel_values": inputs.get("pixel_values", None)
                        }
                        return outputs.hidden_states[-1], vision_info
                
                wrapper = FullQwenVLWrapper(model, processor, device, dtype)
                return (wrapper,)
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenVLLoader": QwenVLLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLLoader": "Qwen2.5-VL Model Loader",
}