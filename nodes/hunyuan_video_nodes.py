"""
HunyuanVideo 1.5 nodes for ComfyUI
Dual text encoder: Qwen2.5-VL + byT5/Glyph-SDXL-v2 (handled by ComfyUI internally)
Vision encoder: SigLIP for I2V
Uses ComfyUI's native HunyuanVideo implementation
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, Tuple, List
import folder_paths

logger = logging.getLogger(__name__)

# Try to import ComfyUI's utilities
try:
    import comfy.sd
    import comfy.model_management as mm
    import comfy.clip_vision
    COMFY_AVAILABLE = True
except ImportError:
    logger.warning("ComfyUI utilities not available")
    COMFY_AVAILABLE = False


class HunyuanVideoCLIPLoader:
    """
    Load HunyuanVideo 1.5 CLIP (Qwen2.5-VL + optional byT5)

    IMPORTANT: Use Qwen2.5-VL-7B-Instruct, NOT Qwen3-VL!

    byT5 handling:
    - ComfyUI automatically handles byT5 encoding when both models are loaded
    - Quoted text ("like this") is automatically extracted and encoded via byT5
    - No separate byT5 output needed - it's integrated into the CLIP object
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get text encoders
        text_encoders = folder_paths.get_filename_list("text_encoders")

        # Filter for Qwen2.5-VL models (NOT Qwen3!)
        qwen_models = [m for m in text_encoders if "qwen" in m.lower() and "2.5" in m.lower()]
        if not qwen_models:
            # Fallback - try any qwen with 7b
            qwen_models = [m for m in text_encoders if "qwen" in m.lower() and "7b" in m.lower()]
        if not qwen_models:
            qwen_models = ["Qwen2.5-VL-7B-Instruct"]  # Default name

        # Filter for byT5 models
        byt5_models = [m for m in text_encoders if "byt5" in m.lower() or "glyph" in m.lower()]

        return {
            "required": {
                "qwen_model": (qwen_models, {
                    "tooltip": "Qwen2.5-VL 7B model (NOT Qwen3!) from models/text_encoders/"
                }),
            },
            "optional": {
                "byt5_model": (["None"] + byt5_models, {
                    "default": "None",
                    "tooltip": "byT5/Glyph model for multilingual text. Put text in quotes for byT5 encoding."
                }),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "HunyuanVideo/Loaders"
    TITLE = "HunyuanVideo CLIP Loader"
    DESCRIPTION = "Load HunyuanVideo 1.5 CLIP. Use Qwen2.5-VL (NOT Qwen3). byT5 handles quoted text automatically."

    def load_clip(
        self,
        qwen_model: str,
        byt5_model: str = "None",
    ) -> Tuple[Any]:
        """
        Load HunyuanVideo CLIP using ComfyUI's native loader.

        byT5 is handled automatically by ComfyUI when both models are loaded.
        The tokenizer extracts quoted text and encodes it via byT5.
        """

        if not COMFY_AVAILABLE:
            raise RuntimeError("ComfyUI not available")

        # Validate Qwen model - warn if Qwen3
        if "qwen3" in qwen_model.lower() or "qwen-3" in qwen_model.lower():
            logger.warning("WARNING: You selected Qwen3 but HunyuanVideo 1.5 needs Qwen2.5-VL!")
            logger.warning("Please use Qwen2.5-VL-7B-Instruct instead.")

        # Get Qwen model path
        qwen_path = folder_paths.get_full_path("text_encoders", qwen_model)
        logger.info(f"Loading Qwen2.5-VL from: {qwen_path}")

        # Build ckpt_paths list
        ckpt_paths = [qwen_path]

        # Add byT5 if selected
        if byt5_model != "None":
            byt5_path = folder_paths.get_full_path("text_encoders", byt5_model)
            if byt5_path and os.path.exists(byt5_path):
                ckpt_paths.append(byt5_path)
                logger.info(f"Loading byT5/Glyph from: {byt5_path}")
                logger.info("byT5 will auto-encode quoted text in prompts")
            else:
                logger.warning(f"byT5 model not found: {byt5_model}")

        # Load using ComfyUI's native HunyuanVideo 1.5 CLIP loader
        # This automatically handles:
        # - Qwen2.5-VL text encoding
        # - byT5 encoding for quoted text
        # - Proper weight routing between models
        clip = comfy.sd.load_clip(
            ckpt_paths=ckpt_paths,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.HUNYUAN_VIDEO_15
        )

        logger.info("Successfully loaded HunyuanVideo 1.5 CLIP")
        if byt5_model != "None":
            logger.info("Tip: Put text in quotes for byT5 multilingual encoding, e.g., 'A sign saying \"Hello World\"'")

        return (clip,)


class HunyuanVideoVisionLoader:
    """
    Load SigLIP vision encoder for HunyuanVideo I2V
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Check for SigLIP models in clip_vision folder
        try:
            vision_models = folder_paths.get_filename_list("clip_vision")
            siglip_models = [m for m in vision_models if "siglip" in m.lower() or "sigclip" in m.lower()]
            if not siglip_models:
                siglip_models = ["sigclip_vision_patch14_384.safetensors"]
        except:
            siglip_models = ["sigclip_vision_patch14_384.safetensors"]

        return {
            "required": {
                "vision_model": (siglip_models, {
                    "tooltip": "SigLIP vision model from models/clip_vision/"
                }),
            }
        }

    RETURN_TYPES = ("CLIP_VISION",)
    RETURN_NAMES = ("clip_vision",)
    FUNCTION = "load_vision"
    CATEGORY = "HunyuanVideo/Loaders"
    TITLE = "HunyuanVideo Vision Loader"
    DESCRIPTION = "Load SigLIP vision encoder for image-to-video"

    def load_vision(self, vision_model: str) -> Tuple[Any]:
        """Load SigLIP vision encoder."""

        if not COMFY_AVAILABLE:
            raise RuntimeError("ComfyUI not available")

        # Get vision model path
        vision_path = folder_paths.get_full_path("clip_vision", vision_model)
        logger.info(f"Loading SigLIP vision from: {vision_path}")

        # Load using ComfyUI's clip_vision loader
        clip_vision = comfy.clip_vision.load(vision_path)

        logger.info("Successfully loaded SigLIP vision encoder")
        return (clip_vision,)


class HunyuanVideoEmptyLatent:
    """
    Create empty latent for HunyuanVideo generation.

    HunyuanVideo uses 32-channel latents (different from standard 4-channel).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 848,
                    "min": 64,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Video width (will be aligned to 16)"
                }),
                "height": ("INT", {
                    "default": 480,
                    "min": 64,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Video height (will be aligned to 16)"
                }),
                "frames": ("INT", {
                    "default": 21,
                    "min": 1,
                    "max": 257,
                    "step": 4,
                    "tooltip": "Number of frames (4n+1 recommended: 21, 45, 69, 93, etc.)"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "HunyuanVideo/Latents"
    TITLE = "HunyuanVideo Empty Latent"
    DESCRIPTION = "Create empty 32-channel latent for HunyuanVideo"

    def generate(
        self,
        width: int,
        height: int,
        frames: int,
        batch_size: int
    ) -> Tuple[Dict]:
        """Generate empty latent tensor for HunyuanVideo."""
        # Align to 16
        width = (width // 16) * 16
        height = (height // 16) * 16

        # HunyuanVideo uses 32 channels, 16x spatial compression, 4x temporal
        latent_channels = 32
        spatial_compression = 16
        temporal_compression = 4

        latent_width = width // spatial_compression
        latent_height = height // spatial_compression
        latent_frames = (frames + temporal_compression - 1) // temporal_compression

        device = mm.intermediate_device()
        latent = torch.zeros(
            [batch_size, latent_channels, latent_frames, latent_height, latent_width],
            device=device
        )

        logger.info(f"Created HunyuanVideo latent: {latent.shape}")

        return ({"samples": latent},)


# Node mappings for __init__.py
NODE_CLASS_MAPPINGS = {
    "HunyuanVideoCLIPLoader": HunyuanVideoCLIPLoader,
    "HunyuanVideoVisionLoader": HunyuanVideoVisionLoader,
    "HunyuanVideoEmptyLatent": HunyuanVideoEmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVideoCLIPLoader": "HunyuanVideo CLIP Loader",
    "HunyuanVideoVisionLoader": "HunyuanVideo Vision Loader (SigLIP)",
    "HunyuanVideoEmptyLatent": "HunyuanVideo Empty Latent",
}
