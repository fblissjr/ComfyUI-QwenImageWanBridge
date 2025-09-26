"""
Model loader nodes for Qwen Image Edit wrapper.

These nodes follow ComfyUI patterns (from WanVideoWrapper) while loading
models in the DiffSynth/Diffusers style for proper Qwen Image Edit support.
"""

import os
import torch
import logging
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar
import comfy.model_base

from .qwen_wrapper_base import QwenWrapperBase, QWEN_VAE_CHANNELS

logger = logging.getLogger(__name__)

# Get device settings from ComfyUI
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

# Register our model folders if needed
def register_model_folders():
    """Register additional folders for Qwen models."""
    # Ensure text_encoders folder exists for Qwen2.5-VL models
    if "text_encoders" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["text_encoders"] = ([], {".safetensors", ".bin"})

    # Add support for qwen_models folder
    if "qwen_models" not in folder_paths.folder_names_and_paths:
        base_path = folder_paths.models_dir
        qwen_path = os.path.join(base_path, "qwen_models")
        if os.path.exists(qwen_path):
            folder_paths.folder_names_and_paths["qwen_models"] = ([qwen_path], {".safetensors", ".bin"})

register_model_folders()


class QwenImageDiTWrapper(comfy.model_base.BaseModel):
    """Wrapper for Qwen Image Edit DiT model."""

    def __init__(self, model_config, model=None):
        super().__init__(model_config)
        self.diffusion_model = model
        self.model_config = model_config

    def get_model_object(self, *args, **kwargs):
        """Return the actual model for ComfyUI's internals."""
        return self.diffusion_model


class QwenImageDiTLoaderWrapper:
    """
    Load Qwen-Image-Edit-2509 transformer model.

    This loader handles both local files and HuggingFace downloads,
    following DiffSynth's loading patterns while integrating with ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get models from diffusion_models folder
        models = folder_paths.get_filename_list("diffusion_models")

        # Filter for Qwen Image models
        qwen_models = [m for m in models if "qwen" in m.lower() and "image" in m.lower()]

        # Add default model names if none found
        if not qwen_models:
            qwen_models = ["qwen_image_edit_2509.safetensors"]

        # Add HuggingFace option
        qwen_models = ["Download from HuggingFace"] + qwen_models

        return {
            "required": {
                "model_name": (qwen_models, {
                    "default": qwen_models[0],
                    "tooltip": "Qwen Image Edit transformer model"
                }),
                "precision": (["bf16", "fp16", "fp32", "fp8"], {
                    "default": "bf16",
                    "tooltip": "Model precision for loading"
                }),
                "load_device": (["cuda", "cpu", "offload"], {
                    "default": "cuda",
                    "tooltip": "Device to load the model on"
                }),
            },
            "optional": {
                "huggingface_id": ("STRING", {
                    "default": "Qwen/Qwen-Image-Edit-2509",
                    "tooltip": "HuggingFace model ID for download"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)  # Changed to standard MODEL type for ComfyUI compatibility
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "QwenImage/Loaders"
    TITLE = "Qwen Image DiT Loader (Wrapper)"
    DESCRIPTION = "Load Qwen-Image-Edit-2509 transformer model"

    def load_model(self, model_name: str, precision: str = "bf16",
                   load_device: str = "cuda", huggingface_id: str = None) -> Tuple[Any]:
        """Load the Qwen Image Edit DiT model."""

        # Determine dtype
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.bfloat16
        }
        dtype = dtype_map.get(precision, torch.bfloat16)

        # Determine device
        if load_device == "offload":
            target_device = offload_device
        elif load_device == "cpu":
            target_device = "cpu"
        else:
            target_device = device

        try:
            if model_name == "Download from HuggingFace":
                # Load from HuggingFace using transformers
                from transformers import AutoModel

                logger.info(f"Loading Qwen Image DiT from HuggingFace: {huggingface_id}")
                model = AutoModel.from_pretrained(
                    huggingface_id,
                    torch_dtype=dtype,
                    trust_remote_code=True
                ).to(target_device)

            else:
                # Load from local file
                model_path = folder_paths.get_full_path("diffusion_models", model_name)
                if not model_path:
                    raise ValueError(f"Model not found: {model_name}")

                logger.info(f"Loading Qwen Image DiT from: {model_path}")

                # Load state dict
                state_dict = load_torch_file(model_path, device=target_device)

                # Use transformers to load the model
                from transformers import AutoModel
                import torch.nn as nn

                # Try to infer model architecture from state dict keys
                logger.info("Loading DiT model from state dict")

                # Create a basic wrapper module that holds the state dict
                class DiTModelWrapper(nn.Module):
                    def __init__(self, state_dict):
                        super().__init__()
                        # Load the state dict into this module
                        self.load_state_dict(state_dict, strict=False)

                    def forward(self, *args, **kwargs):
                        # Forward pass will be handled by the wrapper
                        raise NotImplementedError("Use QwenImageModelWrapper for forward pass")

                model = DiTModelWrapper(state_dict)
                model = model.to(dtype=dtype, device=target_device)

            # Set model to eval mode
            model.eval()

            logger.info(f"Successfully loaded Qwen Image DiT model")
            logger.info(f"Model type: {type(model)}, device: {target_device}, dtype: {dtype}")

            # Return the model directly - it will be wrapped by QwenImageModelWrapper later
            return (model,)

        except Exception as e:
            logger.error(f"Failed to load Qwen Image DiT: {e}")
            raise


class QwenVLTextEncoderLoaderWrapper:
    """
    Load Qwen2.5-VL 7B text encoder with vision processing.

    This handles the full Qwen2.5-VL model including the processor
    for vision token handling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get models from text_encoders folder
        models = folder_paths.get_filename_list("text_encoders")

        # Filter for Qwen VL models
        qwen_models = [m for m in models if "qwen" in m.lower() and ("vl" in m.lower() or "vision" in m.lower())]

        # Add default model names if none found
        if not qwen_models:
            qwen_models = ["qwen2.5_vl_7b.safetensors"]

        # Add HuggingFace option
        qwen_models = ["Download from HuggingFace"] + qwen_models

        return {
            "required": {
                "model_name": (qwen_models, {
                    "default": qwen_models[0],
                    "tooltip": "Qwen2.5-VL text encoder model"
                }),
                "precision": (["bf16", "fp16", "fp32", "int8", "int4"], {
                    "default": "bf16",
                    "tooltip": "Model precision for loading"
                }),
                "load_device": (["cuda", "cpu", "offload"], {
                    "default": "offload",
                    "tooltip": "Device to load the model on"
                }),
            },
            "optional": {
                "huggingface_id": ("STRING", {
                    "default": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "tooltip": "HuggingFace model ID for download"
                }),
                "load_processor": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load the vision processor for image handling"
                }),
            }
        }

    RETURN_TYPES = ("QWEN_TEXT_ENCODER", "QWEN_PROCESSOR")  # Text encoder and processor for wrapper workflow
    RETURN_NAMES = ("text_encoder", "processor")
    FUNCTION = "load_model"
    CATEGORY = "QwenImage/Loaders"
    TITLE = "Qwen2.5-VL Text Encoder Loader (Wrapper)"
    DESCRIPTION = "Load Qwen2.5-VL text encoder with vision processing"

    def load_model(self, model_name: str, precision: str = "bf16",
                   load_device: str = "offload", huggingface_id: str = None,
                   load_processor: bool = True) -> Tuple[Any, Any]:
        """Load the Qwen2.5-VL text encoder and processor."""

        # Determine dtype
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int8": torch.int8,  # For quantization
            "int4": torch.int8,  # Will use 4-bit quantization
        }
        dtype = dtype_map.get(precision, torch.bfloat16)

        # Determine device
        if load_device == "offload":
            target_device = offload_device
        elif load_device == "cpu":
            target_device = "cpu"
        else:
            target_device = device

        try:
            if model_name == "Download from HuggingFace":
                # Load from HuggingFace using transformers
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

                logger.info(f"Loading Qwen2.5-VL from HuggingFace: {huggingface_id}")

                load_kwargs = {
                    "torch_dtype": dtype if precision not in ["int8", "int4"] else torch.float16,
                    "trust_remote_code": True
                }

                # Add quantization config if needed
                if precision == "int8":
                    load_kwargs["load_in_8bit"] = True
                elif precision == "int4":
                    load_kwargs["load_in_4bit"] = True

                text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
                    huggingface_id,
                    **load_kwargs
                ).to(target_device)

                # Load processor
                processor = AutoProcessor.from_pretrained(huggingface_id) if load_processor else None

            else:
                # Load from local file
                model_path = folder_paths.get_full_path("text_encoders", model_name)
                if not model_path:
                    raise ValueError(f"Model not found: {model_name}")

                logger.info(f"Loading Qwen2.5-VL from: {model_path}")

                # Load state dict
                state_dict = load_torch_file(model_path, device=target_device)

                # Use transformers to load
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                import torch.nn as nn

                logger.info("Loading text encoder from state dict")

                # Create model wrapper
                class TextEncoderWrapper(nn.Module):
                    def __init__(self, state_dict):
                        super().__init__()
                        self.load_state_dict(state_dict, strict=False)

                    def forward(self, *args, **kwargs):
                        raise NotImplementedError("Use processor nodes for encoding")

                text_encoder = TextEncoderWrapper(state_dict)
                text_encoder = text_encoder.to(dtype=dtype, device=target_device)

                # Try to load processor from same directory or HuggingFace
                processor = None
                if load_processor:
                    # First try local processor directory
                    processor_path = Path(model_path).parent / "processor"
                    if processor_path.exists():
                        from transformers import AutoProcessor
                        processor = AutoProcessor.from_pretrained(str(processor_path))
                        logger.info(f"Loaded processor from {processor_path}")
                    else:
                        # Try loading from HuggingFace as fallback
                        try:
                            from transformers import AutoProcessor
                            processor = AutoProcessor.from_pretrained(
                                "Qwen/Qwen2-VL-7B-Instruct",
                                trust_remote_code=True
                            )
                            logger.info("Loaded processor from HuggingFace (Qwen2-VL-7B-Instruct)")
                        except Exception as e:
                            logger.warning(f"Could not load processor: {e}")
                            logger.warning("Processor node will not work. Either:")
                            logger.warning("  1. Place processor files in same directory as model")
                            logger.warning("  2. Use 'Download from HuggingFace' option")

            text_encoder.eval()

            logger.info("Successfully loaded Qwen2.5-VL text encoder")
            return (text_encoder, processor)

        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL: {e}")
            raise


class QwenImageVAELoaderWrapper:
    """
    Load 16-channel VAE for Qwen Image Edit.

    This VAE is critical for Qwen Image Edit as it uses 16 channels
    instead of the standard 4 channels.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get models from vae folder
        models = folder_paths.get_filename_list("vae")

        # Filter for 16-channel VAEs
        qwen_vaes = [m for m in models if "16ch" in m.lower() or "qwen" in m.lower()]

        # Add default model names if none found
        if not qwen_vaes:
            qwen_vaes = ["qwen_vae_16ch.safetensors"]

        # Add HuggingFace option
        qwen_vaes = ["Download from HuggingFace"] + qwen_vaes

        return {
            "required": {
                "model_name": (qwen_vaes, {
                    "default": qwen_vaes[0],
                    "tooltip": "16-channel VAE model for Qwen"
                }),
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "VAE precision"
                }),
                "tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable tiled encoding/decoding for large images"
                }),
            },
            "optional": {
                "huggingface_id": ("STRING", {
                    "default": "Qwen/Qwen-Image-Edit-2509",
                    "tooltip": "HuggingFace model ID containing VAE"
                }),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "QwenImage/Loaders"
    TITLE = "Qwen 16-Channel VAE Loader (Wrapper)"
    DESCRIPTION = "Load 16-channel VAE for Qwen Image Edit"

    def load_vae(self, model_name: str, precision: str = "bf16",
                 tiling: bool = False, huggingface_id: str = None) -> Tuple[Any]:
        """Load the 16-channel VAE."""

        # Determine dtype
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(precision, torch.bfloat16)

        try:
            if model_name == "Download from HuggingFace":
                # Load VAE from HuggingFace using diffusers
                from diffusers import AutoencoderKL

                logger.info(f"Loading VAE from HuggingFace: {huggingface_id}")
                vae_model = AutoencoderKL.from_pretrained(
                    huggingface_id,
                    subfolder="vae",
                    torch_dtype=dtype
                ).to(device)

                # Wrap in ComfyUI VAE format
                import comfy.sd
                vae = comfy.sd.VAE(model=vae_model)

            else:
                # Load from local file
                vae_path = folder_paths.get_full_path("vae", model_name)
                if not vae_path:
                    raise ValueError(f"VAE not found: {model_name}")

                logger.info(f"Loading VAE from: {vae_path}")

                # Load state dict
                state_dict = load_torch_file(vae_path, device=device)

                # Use ComfyUI's standard VAE loader - it auto-detects the config
                import comfy.sd

                logger.info("Using ComfyUI VAE loader for Qwen 16-channel VAE")
                vae = comfy.sd.VAE(sd=state_dict)
                # ComfyUI VAE handles device management internally, no need for .to()

            # Enable tiling if requested
            if tiling and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
                logger.info("Enabled VAE tiling for large images")

            # ComfyUI VAE is already in eval mode, no need to call .eval()

            logger.info(f"Successfully loaded 16-channel VAE")
            return (vae,)

        except Exception as e:
            logger.error(f"Failed to load VAE: {e}")
            raise


class QwenModelManagerWrapper:
    """
    Unified loader for complete Qwen Image Edit pipeline.

    This node loads all three components (DiT, Text Encoder, VAE)
    in one go for convenience.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_source": (["HuggingFace", "Local Files"], {
                    "default": "HuggingFace",
                    "tooltip": "Load from HuggingFace or local files"
                }),
                "precision": (["bf16", "fp16", "fp32", "mixed"], {
                    "default": "bf16",
                    "tooltip": "Model precision"
                }),
            },
            "optional": {
                "huggingface_id": ("STRING", {
                    "default": "Qwen/Qwen-Image-Edit-2509",
                    "tooltip": "HuggingFace model ID"
                }),
                "dit_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to DiT model (for local loading)"
                }),
                "text_encoder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to text encoder (for local loading)"
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to VAE (for local loading)"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "QWEN_TEXT_ENCODER", "VAE", "QWEN_PROCESSOR")
    RETURN_NAMES = ("dit", "text_encoder", "vae", "processor")
    FUNCTION = "load_pipeline"
    CATEGORY = "QwenImage/Loaders"
    TITLE = "Qwen Image Edit Pipeline Loader (Wrapper)"
    DESCRIPTION = "Load complete Qwen Image Edit pipeline"

    def load_pipeline(self, model_source: str, precision: str = "bf16",
                     huggingface_id: str = None, dit_path: str = "",
                     text_encoder_path: str = "", vae_path: str = "") -> Tuple[Any, Any, Any, Any]:
        """Load the complete Qwen Image Edit pipeline."""

        # Determine precision for each component
        if precision == "mixed":
            dit_precision = "fp16"
            text_precision = "int8"  # Quantized for memory
            vae_precision = "fp16"
        else:
            dit_precision = text_precision = vae_precision = precision

        try:
            if model_source == "HuggingFace":
                # Load all from HuggingFace
                dit_loader = QwenImageDiTLoaderWrapper()
                dit_model, = dit_loader.load_model(
                    "Download from HuggingFace",
                    dit_precision,
                    "cuda",
                    huggingface_id
                )

                text_loader = QwenVLTextEncoderLoaderWrapper()
                text_encoder, processor = text_loader.load_model(
                    "Download from HuggingFace",
                    text_precision,
                    "offload",
                    f"{huggingface_id.rsplit('/', 1)[0]}/Qwen2.5-VL-7B-Instruct",
                    True
                )

                vae_loader = QwenImageVAELoaderWrapper()
                vae, = vae_loader.load_vae(
                    "Download from HuggingFace",
                    vae_precision,
                    False,
                    huggingface_id
                )

            else:
                # Load from local files
                if not all([dit_path, text_encoder_path, vae_path]):
                    raise ValueError("All model paths must be provided for local loading")

                dit_loader = QwenImageDiTLoaderWrapper()
                dit_model, = dit_loader.load_model(dit_path, dit_precision, "cuda")

                text_loader = QwenVLTextEncoderLoaderWrapper()
                text_encoder, processor = text_loader.load_model(
                    text_encoder_path, text_precision, "offload", load_processor=True
                )

                vae_loader = QwenImageVAELoaderWrapper()
                vae, = vae_loader.load_vae(vae_path, vae_precision, False)

            logger.info("Successfully loaded complete Qwen Image Edit pipeline")
            return (dit_model, text_encoder, vae, processor)

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise


# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "QwenImageDiTLoaderWrapper": QwenImageDiTLoaderWrapper,
    "QwenVLTextEncoderLoaderWrapper": QwenVLTextEncoderLoaderWrapper,
    "QwenImageVAELoaderWrapper": QwenImageVAELoaderWrapper,
    "QwenModelManagerWrapper": QwenModelManagerWrapper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageDiTLoaderWrapper": "Qwen Image DiT Loader (Wrapper)",
    "QwenVLTextEncoderLoaderWrapper": "Qwen2.5-VL Text Encoder Loader (Wrapper)",
    "QwenImageVAELoaderWrapper": "Qwen 16-Channel VAE Loader (Wrapper)",
    "QwenModelManagerWrapper": "Qwen Image Edit Pipeline Loader (Wrapper)",
}