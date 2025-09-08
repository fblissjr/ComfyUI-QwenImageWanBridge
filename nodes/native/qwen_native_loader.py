"""
QwenNativeLoader - Direct Qwen2.5-VL model loading via transformers

This WIP module provides native model loading for Qwen2.5-VL models

Reference implementations:
- DiffSynth-Engine: diffsynth_engine/pipelines/qwen_image.py
- DiffSynth-Studio: diffsynth/pipelines/qwen_image.py
"""

import torch
import logging
from typing import Tuple, Optional, Dict, Any

try:
    from transformers import (
        BitsAndBytesConfig,
        AutoConfig,
        AutoModelForVision2Seq,
        AutoProcessor
    )

    # Try to import Qwen2.5-VL classes (newer) - use AutoProcessor per official docs
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        QWEN25_AVAILABLE = True
    except ImportError as e:
        # Try alternative import path that might work
        try:
            import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen25_module
            Qwen2_5_VLForConditionalGeneration = qwen25_module.Qwen2_5_VLForConditionalGeneration
            QWEN25_AVAILABLE = True
        except Exception:
            QWEN25_AVAILABLE = False

    # Try to import Qwen2-VL classes (older)
    try:
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        QWEN2_AVAILABLE = True
    except ImportError:
        QWEN2_AVAILABLE = False

    TRANSFORMERS_AVAILABLE = True

    # Log transformers version for debugging
    import transformers
    logger = logging.getLogger(__name__)
    logger.info(f"transformers version: {transformers.__version__}")
    logger.info(f"Qwen2.5-VL classes available: {QWEN25_AVAILABLE}")
    logger.info(f"Qwen2-VL classes available: {QWEN2_AVAILABLE}")

    # Try to determine correct class name - might be different
    if not QWEN25_AVAILABLE:
        try:
            # Check if it's available under a different name
            import transformers.models.qwen2_5_vl
            available_classes = [name for name in dir(transformers.models.qwen2_5_vl) if 'Qwen' in name and 'Conditional' in name]
            logger.info(f"Available Qwen2.5-VL classes in transformers: {available_classes}")
        except Exception as e:
            logger.info(f"Could not inspect qwen2_5_vl module: {e}")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers library not available - QwenNativeLoader disabled")

import comfy.model_management as model_management
import comfy.utils
import folder_paths

logger = logging.getLogger(__name__)

class QwenNativeLoader:
    """
    Direct Qwen2.5-VL model loader via transformers

    Features:
    - Direct transformers model loading
    - Custom device placement and memory optimization
    - Quantization support (4bit, 8bit)
    - Low VRAM mode for resource-constrained systems
    - Proper Qwen2VLProcessor integration
    """

    @classmethod
    def INPUT_TYPES(cls):
        if not TRANSFORMERS_AVAILABLE:
            return {
                "required": {
                    "error": ("STRING", {"default": "transformers library required"}),
                }
            }

        # Get local models from text_encoders folder
        models = folder_paths.get_filename_list("text_encoders")
        # Filter for directories (not .safetensors files) - show all directories
        local_models = [m for m in models if not m.endswith('.safetensors')]

        # Add common HuggingFace repo options
        hf_repos = [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL-2B-Instruct"
        ]

        # Combine local and HF options
        all_options = local_models + hf_repos
        if not all_options:
            all_options = ["Qwen/Qwen2.5-VL-7B-Instruct"]

        return {
            "required": {
                "model_path": ("STRING", {
                    "default": all_options[0] if all_options else "Qwen/Qwen2.5-VL-7B-Instruct",
                    "tooltip": "Local directory from 'ComfyUI/models/text_encoders' or HuggingFace repo path. Examples: 'my_qwen_model' (local) or 'Qwen/Qwen2.5-VL-7B-Instruct' (HF repo)"
                }),
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically download from HuggingFace if model not found locally"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device placement. Auto uses ComfyUI's device management"
                }),
                "dtype": (["auto", "fp16", "fp32", "bf16"], {
                    "default": "auto",
                    "tooltip": "Model precision. Auto matches ComfyUI settings"
                }),
                "low_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable low VRAM optimizations (model offloading)"
                }),
                "quantization": (["none", "4bit", "8bit"], {
                    "default": "none",
                    "tooltip": "Quantization for memory reduction. Requires bitsandbytes"
                }),
                "model_variant": (["base", "edit", "edit-distill"], {
                    "default": "edit",
                    "tooltip": "Model variant if multiple available"
                }),
                "trust_remote_code": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow remote code execution (usually required for Qwen)"
                }),
                "cache_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Custom cache directory for model files"
                }),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL", "QWEN_PROCESSOR", "QWEN_CONFIG")
    RETURN_NAMES = ("model", "processor", "config")
    FUNCTION = "load_qwen_native"
    CATEGORY = "QwenImage/Native"
    TITLE = "Qwen Native Loader"
    DESCRIPTION = """
Load Qwen2.5-VL models directly via transformers
"""

    def _get_device_map(self, device: str, low_vram: bool) -> Optional[str]:
        """Get device map based on settings and available memory"""
        if device == "cpu":
            return "cpu"
        elif device == "cuda":
            if low_vram:
                return "auto"  # Let transformers handle offloading
            else:
                return None  # Load entirely on GPU
        else:  # auto
            if low_vram:
                return "auto"
            else:
                return None

    def _get_dtype(self, dtype: str) -> torch.dtype:
        """Get torch dtype from string, respecting ComfyUI settings"""
        if dtype == "auto":
            # Match ComfyUI's model precision
            return model_management.unet_dtype()
        elif dtype == "fp16":
            return torch.float16
        elif dtype == "fp32":
            return torch.float32
        elif dtype == "bf16":
            return torch.bfloat16
        else:
            return torch.float16  # Safe default

    def _get_quantization_config(self, quantization: str) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration if requested"""
        if quantization == "none":
            return None

        try:
            if quantization == "4bit":
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization == "8bit":
                return BitsAndBytesConfig(load_in_8bit=True)
        except Exception as e:
            logger.warning(f"Quantization not available: {e}. Loading without quantization.")
            return None

    def load_qwen_native(
        self,
        model_path: str,
        auto_download: bool = True,
        device: str = "auto",
        dtype: str = "auto",
        low_vram: bool = False,
        quantization: str = "none",
        model_variant: str = "edit",
        trust_remote_code: bool = True,
        cache_dir: str = ""
    ) -> Tuple[Any, Any, Any]:
        """
        Load Qwen2.5-VL model and processor directly from transformers

        Returns:
            Tuple of (model, processor, config) for native processing
        """

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library is required for QwenNativeLoader")

        # Determine if this is a local path or HuggingFace repo
        if "/" in model_path and not model_path.startswith("/"):
            # Looks like HuggingFace repo (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
            if auto_download:
                logger.info(f"Using HuggingFace repo: {model_path}")
                final_model_path = model_path  # Let transformers handle download
            else:
                # Try to find it locally first
                local_path = folder_paths.get_full_path("text_encoders", model_path.split("/")[-1])
                import os
                if os.path.exists(local_path):
                    final_model_path = local_path
                    logger.info(f"Found local copy: {local_path}")
                else:
                    raise RuntimeError(f"Model {model_path} not found locally and auto_download is disabled. "
                                     f"Enable auto_download or manually download to: {local_path}")
        else:
            # Local directory path
            import os
            final_model_path = folder_paths.get_full_path("text_encoders", model_path)
            if not os.path.exists(final_model_path):
                if auto_download and "/" not in model_path:
                    # Try common HF repo format
                    hf_path = f"Qwen/{model_path}"
                    logger.info(f"Local path not found, trying HuggingFace: {hf_path}")
                    final_model_path = hf_path
                else:
                    raise RuntimeError(f"Local model directory not found: {final_model_path}")

        logger.info(f"Loading model from: {final_model_path}")
        logger.info(f"Settings: device={device}, dtype={dtype}, low_vram={low_vram}, quantization={quantization}")

        # Log file structure for debugging
        import os
        if os.path.exists(final_model_path):
            try:
                files = os.listdir(final_model_path)
                logger.info(f"Model directory contains {len(files)} files:")
                for f in sorted(files):
                    if f.endswith(('.json', '.safetensors', '.bin', '.py')):
                        file_path = os.path.join(final_model_path, f)
                        size = os.path.getsize(file_path) / (1024*1024) if os.path.isfile(file_path) else 0
                        logger.info(f"  {f} ({size:.1f} MB)")
            except Exception as list_error:
                logger.warning(f"Could not list directory contents: {list_error}")
        else:
            logger.info(f"Path does not exist locally: {final_model_path}")

        # Prepare loading arguments
        load_kwargs = {
            "dtype": self._get_dtype(dtype),
            "trust_remote_code": trust_remote_code,
        }

        # Device mapping
        device_map = self._get_device_map(device, low_vram)
        if device_map:
            load_kwargs["device_map"] = device_map

        # Quantization
        quantization_config = self._get_quantization_config(quantization)
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            # Don't specify dtype with quantization
            load_kwargs.pop("dtype", None)

        # Cache directory
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        try:
            # Load model - this is the core difference from ComfyUI's approach
            # We load directly instead of going through ComfyUI's CLIP wrapper
            logger.info("Loading Qwen2.5-VL model...")
            logger.info(f"Model path: {final_model_path}")
            logger.info(f"Load kwargs: {load_kwargs}")

            # Try to load config first to verify compatibility
            try:
                config = AutoConfig.from_pretrained(
                    final_model_path,
                    trust_remote_code=trust_remote_code
                )
                logger.info(f"Model config loaded: {config.model_type}")
                logger.info(f"Hidden size: {getattr(config, 'hidden_size', 'unknown')}")
                logger.info(f"Vocab size: {getattr(config, 'vocab_size', 'unknown')}")
                logger.info(f"Architecture: {getattr(config, 'architectures', 'unknown')}")
            except Exception as config_error:
                logger.warning(f"Could not load config: {config_error}")
                config = None

            # Determine correct model class based on config
            model = None
            processor = None
            loading_errors = []

            # Check model type from config to use correct class
            model_type = getattr(config, 'model_type', 'unknown') if config else 'unknown'
            logger.info(f"Detected model type: {model_type}")

            if model_type == 'qwen2_5_vl' and QWEN25_AVAILABLE:
                # Try text encoder model first (like DiffSynth uses) for better conditioning quality
                try:
                    logger.info("Loading Qwen2.5-VL text encoder model with Qwen2_5_VLModel...")
                    from transformers import Qwen2_5_VLModel
                    model = Qwen2_5_VLModel.from_pretrained(
                        final_model_path,
                        **load_kwargs
                    )
                    processor = AutoProcessor.from_pretrained(
                        final_model_path,
                        trust_remote_code=trust_remote_code,
                        cache_dir=cache_dir if cache_dir else None,
                        use_fast=False
                    )
                    logger.info("Successfully loaded Qwen2.5-VL text encoder model")
                except Exception as e0:
                    loading_errors.append(f"Qwen2_5_VLModel: {e0}")
                    logger.warning(f"Failed with Qwen2_5_VLModel: {e0}")
                    
                    # Fallback to conditional generation model
                    try:
                        logger.info("Falling back to Qwen2.5-VL conditional generation model...")
                        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            final_model_path,
                            **load_kwargs
                        )
                        processor = AutoProcessor.from_pretrained(
                            final_model_path,
                            trust_remote_code=trust_remote_code,
                            cache_dir=cache_dir if cache_dir else None,
                            use_fast=False
                        )
                        logger.info("Successfully loaded Qwen2.5-VL model with AutoProcessor")
                    except Exception as e1:
                        loading_errors.append(f"Qwen2_5VLForConditionalGeneration: {e1}")
                        logger.warning(f"Failed with Qwen2_5VLForConditionalGeneration: {e1}")

            elif model_type == 'qwen2_vl' and QWEN2_AVAILABLE:
                # Use Qwen2-VL classes
                try:
                    logger.info("Loading Qwen2-VL model with Qwen2VLForConditionalGeneration...")
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        final_model_path,
                        **load_kwargs
                    )
                    processor = Qwen2VLProcessor.from_pretrained(
                        final_model_path,
                        trust_remote_code=trust_remote_code,
                        cache_dir=cache_dir if cache_dir else None,
                        use_fast=False
                    )
                    logger.info("Successfully loaded Qwen2-VL model and processor")
                except Exception as e2:
                    loading_errors.append(f"Qwen2VLForConditionalGeneration: {e2}")
                    logger.warning(f"Failed with Qwen2VLForConditionalGeneration: {e2}")

            # Fallback to AutoModel if specific classes failed or unavailable
            if model is None:
                try:
                    logger.info("Attempting fallback with AutoModelForVision2Seq...")
                    model = AutoModelForVision2Seq.from_pretrained(
                        final_model_path,
                        **load_kwargs
                    )
                    processor = AutoProcessor.from_pretrained(
                        final_model_path,
                        trust_remote_code=trust_remote_code,
                        cache_dir=cache_dir if cache_dir else None,
                        use_fast=False
                    )
                    logger.info("Successfully loaded with Auto classes")
                except Exception as e3:
                    loading_errors.append(f"AutoModelForVision2Seq: {e3}")
                    logger.warning(f"Failed with AutoModelForVision2Seq: {e3}")

            # If all approaches failed, raise comprehensive error
            if model is None:
                error_details = "\n".join([f"  - {err}" for err in loading_errors])
                raise RuntimeError(f"All model loading approaches failed:\n{error_details}")

            # Processor should already be loaded above with the model

            # Set model to eval mode
            model.eval()

            # Move to device if not using device_map
            if not device_map and device != "cpu":
                target_device = model_management.get_torch_device() if device == "auto" else device
                model = model.to(target_device)
                logger.info(f"Moved model to device: {target_device}")

            # Get configuration for reference
            config = {
                "model_type": model.config.model_type,
                "hidden_size": getattr(model.config, "hidden_size", 3584),
                "vocab_size": getattr(model.config, "vocab_size", 152064),
                "device_map": device_map,
                "dtype": str(model.dtype),
                "quantization": quantization,
                "memory_footprint": self._estimate_memory_footprint(model),
            }

            logger.info("Successfully loaded Qwen native model and processor")
            logger.info(f"Model dtype: {model.dtype}")
            logger.info(f"Estimated memory footprint: {config['memory_footprint']:.1f}GB")

            return (model, processor, config)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load Qwen model: {error_msg}")

            # Provide specific guidance for common errors
            if "size mismatch" in error_msg:
                if "3584" in error_msg and "1280" in error_msg:
                    guidance = (
                        f"Model architecture mismatch detected. "
                        f"The checkpoint appears to be Qwen2.5-VL 7B (3584 dims) but transformers is loading "
                        f"a different model size (1280 dims). This usually means:\n"
                        f"1. Wrong model path - verify '{final_model_path}' contains the correct model\n"
                        f"2. Corrupted model files - try re-downloading\n"
                        f"3. Mixed model files from different variants\n"
                        f"Try using a different model path or re-download the model."
                    )
                else:
                    guidance = (
                        f"Model size mismatch detected. The checkpoint and model architecture don't match. "
                        f"Verify the model path '{final_model_path}' points to a complete, compatible model."
                    )
            elif "trust_remote_code" in error_msg:
                guidance = (
                    f"Remote code execution blocked. Set trust_remote_code=True to load this model. "
                    f"Note: This allows the model to execute custom code."
                )
            else:
                guidance = f"Model loading failed. Check that '{final_model_path}' contains a valid Qwen2.5-VL model."

            raise RuntimeError(f"Model loading failed: {error_msg}\n\nGuidance: {guidance}")

    def _estimate_memory_footprint(self, model) -> float:
        """Estimate model memory footprint in GB"""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            bytes_per_param = 2 if model.dtype == torch.float16 else 4
            memory_gb = (total_params * bytes_per_param) / (1024**3)
            return memory_gb
        except:
            return 0.0  # Fallback if estimation fails

# Node registration will be handled in __init__.py
NODE_CLASS_MAPPINGS = {
    "QwenNativeLoader": QwenNativeLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenNativeLoader": "Qwen Native Loader"
}
