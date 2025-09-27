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

# Set up verbose logging for wrapper nodes
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)

# Add console handler if not present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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
        """Load the Qwen Image Edit DiT model using DiffSynth implementation."""

        logger.info("="*60)
        logger.info("QWEN IMAGE DIT LOADER (DiffSynth) - Starting")
        logger.info(f"Model: {model_name}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Load device: {load_device}")
        logger.info("="*60)

        # Determine dtype
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.bfloat16
        }
        dtype = dtype_map.get(precision, torch.bfloat16)
        logger.debug(f"Using dtype: {dtype}")

        # Determine device
        if load_device == "offload":
            target_device = offload_device
        elif load_device == "cpu":
            target_device = "cpu"
        else:
            target_device = device
        logger.debug(f"Target device: {target_device}")

        try:
            # Always use DiffSynth's QwenImageDiT implementation
            import sys
            import os
            # Add parent directory to path to import from models folder
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            from models.qwen_image_dit import QwenImageDiT

            logger.info("Creating QwenImageDiT (DiffSynth implementation)")
            model = QwenImageDiT()

            if model_name == "Download from HuggingFace":
                # Download state dict from HuggingFace
                logger.info(f"Downloading state dict from HuggingFace: {huggingface_id}")

                from huggingface_hub import snapshot_download
                import tempfile

                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Download the model files
                    snapshot_download(
                        repo_id=huggingface_id,
                        local_dir=tmp_dir,
                        local_dir_use_symlinks=False,
                        allow_patterns=["transformer/*.safetensors", "transformer/*.bin"],
                    )

                    # Find the model file in transformer subfolder
                    model_files = []
                    transformer_dir = os.path.join(tmp_dir, "transformer")
                    if os.path.exists(transformer_dir):
                        for filename in os.listdir(transformer_dir):
                            if filename.endswith((".safetensors", ".bin")):
                                model_files.append(os.path.join(transformer_dir, filename))

                    if not model_files:
                        # Fallback to root directory
                        for filename in os.listdir(tmp_dir):
                            if filename.endswith((".safetensors", ".bin")):
                                model_files.append(os.path.join(tmp_dir, filename))

                    if not model_files:
                        raise ValueError(f"No model files found in {huggingface_id}")

                    logger.info(f"Found model files: {model_files}")

                    # Load state dict from files (may be multiple)
                    state_dict = {}
                    for model_file in model_files:
                        file_state_dict = load_torch_file(model_file, device=target_device)
                        state_dict.update(file_state_dict)

            else:
                # Load from local file
                model_path = folder_paths.get_full_path("diffusion_models", model_name)
                if not model_path:
                    raise ValueError(f"Model not found: {model_name}")

                logger.info(f"Loading Qwen Image DiT from: {model_path}")

                # Load state dict
                state_dict = load_torch_file(model_path, device=target_device)

            # Apply state dict converter if model has one
            if hasattr(model, 'state_dict_converter'):
                converter = model.state_dict_converter()
                if converter:
                    logger.info("Applying state dict converter")
                    state_dict = converter.from_diffusers(state_dict)

            # Load the state dict into the model
            logger.info("Loading state dict into QwenImageDiT")
            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                logger.warning(f"Missing keys: {len(incompatible.missing_keys)}")
                if len(incompatible.missing_keys) < 10:
                    logger.debug(f"Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.warning(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
                if len(incompatible.unexpected_keys) < 10:
                    logger.debug(f"Unexpected keys: {incompatible.unexpected_keys}")

            # Move model to target device and dtype
            model = model.to(dtype=dtype, device=target_device)
            model.eval()

            logger.info(f"Successfully loaded Qwen Image DiT model")
            logger.info(f"Model type: {type(model).__name__}, device: {target_device}, dtype: {dtype}")

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
        """Load the Qwen2.5-VL text encoder and processor using DiffSynth implementation."""

        logger.info("="*60)
        logger.info("QWEN2.5-VL TEXT ENCODER LOADER (DiffSynth) - Starting")
        logger.info(f"Model: {model_name}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Load device: {load_device}")
        logger.info("="*60)

        # Determine dtype
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int8": torch.int8,  # For quantization
            "int4": torch.int8,  # Will use 4-bit quantization
        }
        dtype = dtype_map.get(precision, torch.bfloat16)
        logger.debug(f"Using dtype: {dtype}")

        # Determine device
        if load_device == "offload":
            target_device = offload_device
        elif load_device == "cpu":
            target_device = "cpu"
        else:
            target_device = device
        logger.debug(f"Target device: {target_device}")

        try:
            # Always use DiffSynth's QwenImageTextEncoder implementation
            import sys
            import os
            # Add parent directory to path to import from models folder
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            from models.qwen_image_text_encoder import QwenImageTextEncoder

            logger.info("Creating QwenImageTextEncoder (DiffSynth implementation)")
            text_encoder = QwenImageTextEncoder()

            if model_name == "Download from HuggingFace":
                # Download state dict from HuggingFace
                logger.info(f"Downloading state dict from HuggingFace: {huggingface_id}")

                from huggingface_hub import snapshot_download
                import tempfile

                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Download the model files
                    snapshot_download(
                        repo_id=huggingface_id,
                        local_dir=tmp_dir,
                        local_dir_use_symlinks=False,
                        allow_patterns=["*.safetensors", "*.bin"],
                    )

                    # Find the model file
                    model_files = []
                    for filename in os.listdir(tmp_dir):
                        if filename.endswith((".safetensors", ".bin")):
                            model_files.append(os.path.join(tmp_dir, filename))

                    if not model_files:
                        raise ValueError(f"No model files found in {huggingface_id}")

                    logger.info(f"Found model files: {model_files}")

                    # Load state dict from the first file
                    state_dict = load_torch_file(model_files[0], device=target_device)

                # Apply state dict converter if needed
                converter = text_encoder.state_dict_converter()
                if converter:
                    logger.info("Applying state dict converter")
                    state_dict = converter.from_diffusers(state_dict)

                # Load processor separately if requested
                processor = None
                if load_processor:
                    try:
                        # Use the specific Qwen2_5_VLProcessor instead of AutoProcessor
                        from transformers import Qwen2_5_VLProcessor
                        processor = Qwen2_5_VLProcessor.from_pretrained(
                            huggingface_id,
                            trust_remote_code=True
                        )
                        logger.info(f"Loaded processor: {type(processor).__name__}")
                    except ImportError:
                        # Fallback to AutoProcessor if Qwen2_5_VLProcessor not available
                        logger.info("Qwen2_5_VLProcessor not available, using AutoProcessor")
                        from transformers import AutoProcessor
                        processor = AutoProcessor.from_pretrained(
                            huggingface_id,
                            trust_remote_code=True
                        )
                        logger.info(f"Loaded processor: {type(processor).__name__}")
                    except Exception as e:
                        logger.warning(f"Failed to load processor: {e}")
                        processor = None

            else:
                # Load from local file using DiffSynth approach
                model_path = folder_paths.get_full_path("text_encoders", model_name)
                if not model_path:
                    raise ValueError(f"Model not found: {model_name}")

                logger.info(f"Loading Qwen2.5-VL from: {model_path}")

                # Load state dict from file
                state_dict = load_torch_file(model_path, device=target_device)

                # Apply state dict converter if needed
                converter = text_encoder.state_dict_converter()
                if converter:
                    logger.info("Applying state dict converter")
                    state_dict = converter.from_diffusers(state_dict)

                # Load processor if requested
                processor = None
                if load_processor:
                    # Try to find processor config in various locations
                    model_path_obj = Path(model_path)
                    model_dir = model_path_obj.parent if model_path_obj.is_file() else model_path_obj

                    # Try different locations for processor config
                    processor_locations = [
                        model_dir,  # Same directory as model
                        model_dir / "processor",  # processor subdirectory
                        Path(__file__).parent.parent / "configs" / "qwen25vl",  # Local configs
                    ]

                    for proc_path in processor_locations:
                        # Check for both preprocessor_config.json and processor_config.json
                        config_file = None
                        if proc_path.exists():
                            if (proc_path / "preprocessor_config.json").exists():
                                config_file = "preprocessor_config.json"
                            elif (proc_path / "processor_config.json").exists():
                                config_file = "processor_config.json"

                        if config_file:
                            try:
                                # Try Qwen2_5_VLProcessor first
                                from transformers import Qwen2_5_VLProcessor
                                processor = Qwen2_5_VLProcessor.from_pretrained(
                                    str(proc_path),
                                    trust_remote_code=True
                                )
                                logger.info(f"Loaded Qwen2_5_VLProcessor from {proc_path}")
                                break
                            except ImportError:
                                # Fallback to AutoProcessor
                                from transformers import AutoProcessor
                                processor = AutoProcessor.from_pretrained(
                                    str(proc_path),
                                    trust_remote_code=True
                                )
                                logger.info(f"Loaded AutoProcessor from {proc_path}")
                                break
                            except Exception as e:
                                logger.debug(f"Could not load processor from {proc_path}: {e}")
                                continue

                    # Fallback to HuggingFace if no local processor found
                    if processor is None:
                        try:
                            # Try specific processor first
                            from transformers import Qwen2_5_VLProcessor
                            processor = Qwen2_5_VLProcessor.from_pretrained(
                                "Qwen/Qwen2.5-VL-7B-Instruct",
                                trust_remote_code=True
                            )
                            logger.info("Loaded Qwen2_5_VLProcessor from HuggingFace")
                        except ImportError:
                            # Fallback to AutoProcessor
                            from transformers import AutoProcessor
                            processor = AutoProcessor.from_pretrained(
                                "Qwen/Qwen2.5-VL-7B-Instruct",
                                trust_remote_code=True
                            )
                            logger.info("Loaded AutoProcessor from HuggingFace")
                        except Exception as e:
                            logger.warning(f"Could not load processor: {e}")
                            logger.warning("Processor will not be available")

            # Load the state dict into the model
            logger.info("Loading state dict into QwenImageTextEncoder")
            incompatible = text_encoder.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                logger.warning(f"Missing keys: {len(incompatible.missing_keys)}")
                if len(incompatible.missing_keys) < 10:
                    logger.debug(f"Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.warning(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
                if len(incompatible.unexpected_keys) < 10:
                    logger.debug(f"Unexpected keys: {incompatible.unexpected_keys}")

            # Move model to target device and dtype
            text_encoder = text_encoder.to(dtype=dtype, device=target_device)
            text_encoder.eval()

            logger.info(f"Successfully loaded Qwen2.5-VL text encoder")
            logger.info(f"Model on device: {target_device}, dtype: {dtype}")
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

        logger.info("="*60)
        logger.info("QWEN 16-CHANNEL VAE LOADER - Starting")
        logger.info(f"Model: {model_name}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Tiling: {tiling}")
        logger.info("="*60)

        # Determine dtype
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(precision, torch.bfloat16)
        logger.debug(f"Using dtype: {dtype}")

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

                # Detect VAE channels following Kijai's pattern
                if "encoder.conv_in.weight" in state_dict:
                    in_channels = state_dict["encoder.conv_in.weight"].shape[1]
                    logger.info(f"Detected VAE input channels: {in_channels}")
                elif "model.encoder.conv_in.weight" in state_dict:
                    in_channels = state_dict["model.encoder.conv_in.weight"].shape[1]
                    logger.info(f"Detected VAE input channels (nested): {in_channels}")
                else:
                    logger.warning("Could not detect VAE input channels, assuming 16")
                    in_channels = 16

                # Check output channels
                if "decoder.conv_out.weight" in state_dict:
                    out_channels = state_dict["decoder.conv_out.weight"].shape[0]
                    logger.info(f"Detected VAE output channels: {out_channels}")
                elif "model.decoder.conv_out.weight" in state_dict:
                    out_channels = state_dict["model.decoder.conv_out.weight"].shape[0]
                    logger.info(f"Detected VAE output channels (nested): {out_channels}")
                else:
                    logger.warning("Could not detect VAE output channels, assuming 16")
                    out_channels = 16

                # Validate this is a 16-channel VAE
                if in_channels != 3 or out_channels != 3:
                    logger.warning(f"Non-standard VAE detected: in={in_channels}, out={out_channels}")
                    logger.warning("This appears to be a special VAE (possibly 16-channel latent space)")

                # Check latent channels (most important for Qwen)
                if "decoder.conv_in.weight" in state_dict:
                    latent_channels = state_dict["decoder.conv_in.weight"].shape[1]
                    logger.info(f"Detected VAE latent channels: {latent_channels}")
                    if latent_channels == 16:
                        logger.info("✓ Confirmed: This is a 16-channel latent VAE (Qwen/Wan format)")
                elif "model.decoder.conv_in.weight" in state_dict:
                    latent_channels = state_dict["model.decoder.conv_in.weight"].shape[1]
                    logger.info(f"Detected VAE latent channels (nested): {latent_channels}")
                    if latent_channels == 16:
                        logger.info("✓ Confirmed: This is a 16-channel latent VAE (Qwen/Wan format)")

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