"""
Unified Qwen2.5-VL Implementation
Replaces ComfyUI CLIP with our own transformers-based implementation
Supports local models and external API endpoints
"""

import os
import json
import logging
import torch
from typing import Optional, Dict, Any, Tuple, Union, List
import folder_paths

logger = logging.getLogger(__name__)

# Try to import transformers for local model support
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - only API mode will work")

# Try to import requests for API support
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests not available - only local models will work")


class QwenUnifiedModel:
    """
    Unified Qwen2.5-VL model that supports local transformers with quantization and external APIs
    """
    
    def __init__(self, model_path: str = None, api_endpoint: str = None, api_key: str = None,
                 torch_dtype: str = "float16", quantization: str = "none", device_map: str = "auto",
                 flash_attention: bool = True):
        self.model_path = model_path
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.torch_dtype = torch_dtype
        self.quantization = quantization
        self.device_map = device_map
        self.flash_attention = flash_attention
        
        # Model components (for local mode)
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Mode tracking
        self.is_local = model_path is not None
        self.is_api = api_endpoint is not None
        
        # Custom paths for Qwen Image Edit structure
        self.custom_tokenizer_path = None
        self.custom_processor_path = None
        
        if self.is_local:
            self._load_local_model()
        elif self.is_api:
            self._setup_api()
        else:
            raise ValueError("Must provide either model_path or api_endpoint")
    
    def _load_local_model(self):
        """Load local transformers model with quantization support"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not available for local model loading")
        
        logger.info(f"Loading Qwen2.5-VL from: {self.model_path}")
        logger.info(f"Configuration: dtype={self.torch_dtype}, quantization={self.quantization}, device_map={self.device_map}")
        
        try:
            import os
            
            # Check for unsupported single safetensors files
            if self.model_path.endswith(".safetensors"):
                raise ValueError(
                    f"Single safetensors files not supported in unified loader.\n"
                    f"For single safetensors files, use QwenVLCLIPLoader + QwenVLTextEncoder.\n" 
                    f"The unified loader is for complete model directories or API endpoints."
                )
            
            # Check if it's a directory or HuggingFace model ID
            if os.path.isdir(self.model_path):
                # It's a directory - load with transformers
                logger.info("Loading from directory with transformers")
                
                # Check if this is a Qwen Image Edit model structure
                parent_dir = os.path.dirname(self.model_path) if self.model_path.endswith(('text_encoder', 'tokenizer', 'processor')) else self.model_path
                
                # Look for Qwen Image Edit structure (separate folders)
                text_encoder_dir = os.path.join(parent_dir, 'text_encoder')
                tokenizer_dir = os.path.join(parent_dir, 'tokenizer') 
                processor_dir = os.path.join(parent_dir, 'processor')
                
                if os.path.exists(text_encoder_dir) and os.path.exists(tokenizer_dir) and os.path.exists(processor_dir):
                    logger.info("Detected Qwen Image Edit model structure with separate folders")
                    model_path_to_use = text_encoder_dir
                    self.custom_tokenizer_path = tokenizer_dir
                    self.custom_processor_path = processor_dir
                else:
                    # Standard HuggingFace model directory
                    model_path_to_use = self.model_path
                    self.custom_tokenizer_path = None
                    self.custom_processor_path = None
                
                local_files_only = True
                
            else:
                # Assume it's a HuggingFace model ID
                logger.info(f"Loading as HuggingFace model ID: {self.model_path}")
                model_path_to_use = self.model_path
                local_files_only = False
            
            # Prepare model loading arguments
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": self.device_map if self.device_map != "none" else None,
                "local_files_only": local_files_only
            }
            
            # Handle torch_dtype
            if self.torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif self.torch_dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif self.torch_dtype == "float32":
                model_kwargs["torch_dtype"] = torch.float32
            
            # Handle quantization
            if self.quantization == "8bit":
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_skip_modules=["visual"]  # Don't quantize vision components
                    )
                    logger.info("Enabled 8-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not available, falling back to float16")
                    model_kwargs["torch_dtype"] = torch.float16
            
            elif self.quantization == "4bit":
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_skip_modules=["visual"]  # Don't quantize vision components
                    )
                    logger.info("Enabled 4-bit quantization with NF4")
                except ImportError:
                    logger.warning("bitsandbytes not available, falling back to float16")
                    model_kwargs["torch_dtype"] = torch.float16
            
            # Handle flash attention
            if self.flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Enabled Flash Attention 2")
            
            # Load processor and model
            # Use custom processor path if available (for Qwen Image Edit)
            processor_path = self.custom_processor_path or model_path_to_use
            self.processor = AutoProcessor.from_pretrained(
                processor_path, 
                trust_remote_code=True,
                local_files_only=local_files_only
            )
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path_to_use,
                **model_kwargs
            )
            
            # Load tokenizer - use custom path if available
            if self.custom_tokenizer_path:
                logger.info(f"Loading custom tokenizer from: {self.custom_tokenizer_path}")
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.custom_tokenizer_path,
                    trust_remote_code=True,
                    local_files_only=local_files_only
                )
                # Verify custom tokens are loaded
                if hasattr(self.tokenizer, 'added_tokens_encoder'):
                    custom_tokens = list(self.tokenizer.added_tokens_encoder.keys())
                    logger.info(f"Loaded {len(custom_tokens)} custom tokens including spatial reference tokens")
            else:
                self.tokenizer = self.processor.tokenizer
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
            
            logger.info("Successfully loaded local Qwen2.5-VL model")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def _get_torch_dtype(self):
        """Convert string dtype to torch dtype"""
        if self.torch_dtype == "float16":
            return torch.float16
        elif self.torch_dtype == "bfloat16":
            return torch.bfloat16
        elif self.torch_dtype == "float32":
            return torch.float32
        else:
            return torch.float16  # default
    
    def _setup_api(self):
        """Setup API connection"""
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("Requests not available for API mode")
        
        logger.info(f"Setting up API connection to: {self.api_endpoint}")
        
        # Test API connection
        try:
            response = requests.get(f"{self.api_endpoint}/v1/models", 
                                  headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                                  timeout=10)
            if response.status_code == 200:
                logger.info("API connection successful")
            else:
                logger.warning(f"API connection test returned {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not test API connection: {e}")
    
    def tokenize(self, text: str, images: Optional[List] = None) -> Dict[str, Any]:
        """
        Tokenize text and images for ComfyUI compatibility
        Returns token structure compatible with our encoding pipeline
        """
        if self.is_local:
            return self._tokenize_local(text, images)
        else:
            # For API mode, we'll handle this differently in generate/encode
            return {"text": text, "images": images, "mode": "api"}
    
    def _tokenize_local(self, text: str, images: Optional[List] = None) -> Dict[str, Any]:
        """Tokenize using local model"""
        
        # Convert images to PIL if needed
        pil_images = None
        if images is not None:
            pil_images = self._convert_images_to_pil(images)
        
        # Process inputs
        inputs = self.processor(
            text=text,
            images=pil_images,
            return_tensors="pt"
        )
        
        # Return in ComfyUI-compatible format
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "mode": "local"
        }
    
    def encode_from_tokens(self, tokens: Dict[str, Any]) -> torch.Tensor:
        """
        Encode tokens to conditioning embeddings
        Compatible with ComfyUI's conditioning system
        """
        if tokens.get("mode") == "local" and self.is_local:
            return self._encode_local(tokens)
        elif tokens.get("mode") == "api" and self.is_api:
            # For API mode, we'll return a placeholder that gets resolved during generation
            return self._encode_api_placeholder(tokens)
        else:
            raise ValueError(f"Incompatible token mode {tokens.get('mode')} with model mode")
    
    def _encode_local(self, tokens: Dict[str, Any]) -> torch.Tensor:
        """Encode using local model"""
        
        with torch.no_grad():
            # Check if this is a safetensors wrapper (no actual model to call)
            if hasattr(self.model, 'is_safetensors_only') and self.model.is_safetensors_only:
                # Use the wrapper's encode_from_tokens method
                return self.model.encode_from_tokens(tokens)
            
            # Full model - use standard transformers approach
            inputs = {k: v for k, v in tokens.items() if k != "mode" and v is not None}
            
            # Move to model device
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Get embeddings from the model
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            
            # Use the last hidden state as conditioning
            conditioning = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
            
            return conditioning
    
    def _encode_api_placeholder(self, tokens: Dict[str, Any]) -> torch.Tensor:
        """Get actual embeddings from API for conditioning"""
        text = tokens.get("text", "")
        images = tokens.get("images")
        
        try:
            # Call heylookitsanllm embeddings endpoint
            embeddings = self.get_embeddings_api(text, images)
            logger.info(f"Retrieved embeddings from API: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.warning(f"API embeddings failed, using placeholder: {e}")
            # Fallback to placeholder
            return torch.zeros(1, 1, 3584, dtype=torch.float32)  # 3584 is Qwen2.5-VL hidden size
    
    def generate(self, text: str, images: Optional[List] = None, **kwargs) -> str:
        """
        Generate text completion
        Used for conversation generation in Template Builder
        """
        if self.is_local:
            return self._generate_local(text, images, **kwargs)
        else:
            return self._generate_api(text, images, **kwargs)
    
    def _generate_local(self, text: str, images: Optional[List] = None, **kwargs) -> str:
        """Generate using local model"""
        
        # Convert images to PIL if needed
        pil_images = None
        if images is not None:
            pil_images = self._convert_images_to_pil(images)
        
        # Process inputs
        inputs = self.processor(
            text=text,
            images=pil_images,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate
        generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "do_sample": kwargs.get("do_sample", True),
            "top_p": kwargs.get("top_p", 0.9),
            "pad_token_id": self.processor.tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        # Decode only the new tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        return generated_text
    
    def _generate_api(self, text: str, images: Optional[List] = None, **kwargs) -> str:
        """Generate text using external API (for conversation generation)"""
        return self._call_api("generate", text, images, **kwargs)
    
    def get_embeddings_api(self, text: str, images: Optional[List] = None) -> torch.Tensor:
        """Get embeddings using external API (for conditioning)"""
        return self._call_api("embeddings", text, images)
    
    def _call_api(self, endpoint_type: str, text: str, images: Optional[List] = None, **kwargs):
        """Generic API caller that handles different endpoint types"""
        
        # Prepare request headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Convert images to base64 if needed
        image_data = None
        if images is not None:
            image_data = self._convert_images_to_base64(images)
        
        # Prepare payload based on endpoint type
        if endpoint_type == "generate":
            # Text generation (OpenAI chat completions format)
            payload = {
                "model": "qwen2.5-vl",
                "messages": [{"role": "user", "content": text}],
                "max_tokens": kwargs.get("max_new_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9)
            }
            
            # Add images if present
            if image_data:
                payload["messages"][0]["images"] = image_data
            
            endpoint_url = f"{self.api_endpoint}/v1/chat/completions"
            
        elif endpoint_type == "embeddings":
            # Embedding generation (OpenAI embeddings format)
            payload = {
                "model": "qwen2.5-vl-embedding",  # Different model for embeddings
                "input": text,
                "encoding_format": "float"
            }
            
            # Add images if present (custom extension)
            if image_data:
                payload["images"] = image_data
            
            endpoint_url = f"{self.api_endpoint}/v1/embeddings"
            
        else:
            raise ValueError(f"Unknown endpoint type: {endpoint_type}")
        
        try:
            response = requests.post(
                endpoint_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if endpoint_type == "generate":
                    return result["choices"][0]["message"]["content"]
                    
                elif endpoint_type == "embeddings":
                    # Extract embeddings and convert to tensor
                    embeddings = result["data"][0]["embedding"]
                    return torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
                    
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                raise RuntimeError(f"API request failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def _convert_images_to_pil(self, images):
        """Convert various image formats to PIL Images"""
        import PIL.Image
        import numpy as np
        
        if images is None:
            return None
        
        pil_images = []
        
        for img in images if isinstance(images, list) else [images]:
            if isinstance(img, torch.Tensor):
                # Convert tensor to PIL
                if img.dim() == 4:  # Batch dimension
                    img = img[0]  # Take first image
                
                img_array = img.cpu().numpy()
                
                # Normalize if needed
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
                
                # Ensure RGB format
                if img_array.shape[-1] == 3:
                    pil_img = PIL.Image.fromarray(img_array)
                else:
                    # Convert from HWC to HWC if needed
                    pil_img = PIL.Image.fromarray(img_array)
                
                pil_images.append(pil_img)
            elif isinstance(img, PIL.Image.Image):
                pil_images.append(img)
            else:
                logger.warning(f"Unknown image format: {type(img)}")
        
        return pil_images if len(pil_images) > 1 else pil_images[0] if pil_images else None
    
    def _convert_images_to_base64(self, images):
        """Convert images to base64 for API requests"""
        import base64
        from io import BytesIO
        
        if images is None:
            return None
        
        base64_images = []
        pil_images = self._convert_images_to_pil(images)
        
        if not isinstance(pil_images, list):
            pil_images = [pil_images]
        
        for pil_img in pil_images:
            buffer = BytesIO()
            pil_img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            base64_images.append(f"data:image/png;base64,{img_base64}")
        
        return base64_images


class QwenUnifiedLoader:
    """
    Unified loader that replaces QwenVLCLIPLoader
    Supports local models with quantization and external APIs
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("text_encoders")
        # Filter for directories only (no single safetensors files)
        qwen_models = [m for m in models if "qwen" in m.lower() and not m.endswith(".safetensors")]
        if not qwen_models:
            qwen_models = ["Qwen2.5-VL-7B", "Qwen/Qwen2.5-VL-7B"]
        
        return {
            "required": {
                "model_name": (qwen_models, {
                    "tooltip": "Complete Qwen2.5-VL model directory or HuggingFace ID. For single .safetensors files, use QwenVLCLIPLoader instead."
                }),
                "torch_dtype": (["float16", "bfloat16", "float32"], {
                    "default": "float16",
                    "tooltip": "Model precision: float16=memory efficient, bfloat16=stable, float32=high precision"
                }),
                "quantization": (["none", "8bit", "4bit"], {
                    "default": "none",
                    "tooltip": "Quantization: none=full precision, 8bit=half memory, 4bit=quarter memory"
                }),
                "device_map": (["auto", "cuda", "cpu", "none"], {
                    "default": "auto",
                    "tooltip": "Device placement: auto=automatic, cuda=GPU only, cpu=CPU only, none=manual"
                }),
            },
            "optional": {
                "flash_attention": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Flash Attention 2 for faster inference (requires compatible GPU)"
                }),
                "api_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable API mode instead of local model"
                }),
                "api_endpoint": ("STRING", {
                    "default": "http://localhost:8000",
                    "tooltip": "API endpoint URL (only used if api_mode=True)"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key if required (only used if api_mode=True)"
                }),
            }
        }
    
    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load_model"
    CATEGORY = "QwenImage/Loaders"
    TITLE = "Qwen2.5-VL Unified Loader"
    DESCRIPTION = "Load complete Qwen2.5-VL models (directories/HF IDs) or external APIs. For single .safetensors files, use QwenVLCLIPLoader."
    
    def load_model(self, model_name: str, torch_dtype: str = "float16", 
                  quantization: str = "none", device_map: str = "auto",
                  flash_attention: bool = True, api_mode: bool = False,
                  api_endpoint: str = None, api_key: str = None) -> Tuple[QwenUnifiedModel]:
        """Load Qwen2.5-VL model with specified configuration"""
        
        if api_mode:
            if not api_endpoint:
                raise ValueError("api_endpoint required when api_mode=True")
            
            logger.info(f"Connecting to Qwen2.5-VL API at: {api_endpoint}")
            
            qwen_model = QwenUnifiedModel(
                api_endpoint=api_endpoint,
                api_key=api_key if api_key else None
            )
            
        else:
            # Local mode (default)
            if not model_name:
                raise ValueError("model_name required for local mode")
            
            model_path = folder_paths.get_full_path("text_encoders", model_name)
            logger.info(f"Loading local Qwen2.5-VL from: {model_path}")
            logger.info(f"Configuration: {torch_dtype}, quantization={quantization}, device_map={device_map}")
            
            qwen_model = QwenUnifiedModel(
                model_path=model_path,
                torch_dtype=torch_dtype,
                quantization=quantization,
                device_map=device_map,
                flash_attention=flash_attention
            )
        
        return (qwen_model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenUnifiedLoader": QwenUnifiedLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenUnifiedLoader": "Qwen2.5-VL Unified Loader",
}