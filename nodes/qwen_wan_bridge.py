"""
Qwen-to-Wan Video Bridge Nodes

Bridge nodes for connecting Qwen-Image-Edit output to Wan Video models.
Enables image-to-video workflows with Qwen editing as first frame.

Architecture:
- Qwen-Image-Edit generates high-quality first frame
- Wan Video extends to temporal sequence
- Both use 16-channel Wan VAE latents (shared latent space)
"""

import torch
import logging
from typing import Optional, Tuple, Any
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import comfy.utils
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    logger.warning("ComfyUI utilities not available")


class QwenToWanFirstFrameLatent:
    """
    Bridge node: Prepare Qwen output as first frame latent for Wan Video.

    Takes Qwen-Image-Edit output and encodes it to 16-channel latent
    with temporal dimension added for Wan Video compatibility.

    Usage:
        Qwen Pipeline → This Node → Wan Video Pipeline
        (via external tools or future ComfyUI nodes)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Image from Qwen-Image-Edit (edited first frame)"
                }),
                "vae": ("VAE", {
                    "tooltip": "16-channel Wan VAE (qwen_image_vae.safetensors or compatible)"
                }),
            },
            "optional": {
                "bridge_mode": (["wan_video", "chronoedit"], {
                    "default": "wan_video",
                    "tooltip": "wan_video: Full normalization for DiffSynth | chronoedit: Simple encoding for Kijai's nodes"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply Wan normalization (mean/std). Auto-disabled for chronoedit mode."
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show shape and normalization details"
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("first_frame_latent", "debug_info")
    FUNCTION = "encode_first_frame"
    CATEGORY = "QwenWanBridge"
    TITLE = "Qwen to Wan First Frame Latent"
    DESCRIPTION = "Encode Qwen output as first frame latent for Wan Video (16-channel, temporal)"

    def encode_first_frame(self, image: torch.Tensor, vae, bridge_mode: str = "wan_video",
                          normalize: bool = True, debug_mode: bool = False) -> Tuple[Any, str]:
        """
        Encode Qwen image output to Wan Video compatible first frame latent.

        Args:
            image: Qwen output image [B, H, W, C]
            vae: 16-channel Wan VAE encoder
            bridge_mode: "wan_video" for DiffSynth, "chronoedit" for Kijai's nodes
            normalize: Apply Wan Video normalization (auto-disabled for chronoedit)
            debug_mode: Show processing details

        Returns:
            (first_frame_latent, debug_info)
        """
        debug_info = []

        if not COMFY_AVAILABLE:
            raise RuntimeError("ComfyUI not available")

        # Input validation
        if image.shape[0] != 1:
            raise ValueError(f"Expected single image (batch=1), got batch={image.shape[0]}")

        # ChronoEdit mode: simpler processing for Kijai's WanVideoImageToVideoEncode
        if bridge_mode == "chronoedit":
            normalize = False  # Kijai's nodes handle normalization internally
            debug_info.append("Mode: ChronoEdit (simple encoding for Kijai's nodes)")
        else:
            debug_info.append("Mode: Wan Video (DiffSynth normalization)")

        original_shape = image.shape
        debug_info.append(f"Input image shape: {original_shape}")

        # Encode using VAE
        # ComfyUI format: [B, H, W, C] → VAE expects [B, H, W, 3]
        latent = vae.encode(image[:, :, :, :3])

        # Check latent channels
        if latent.shape[1] != 16:
            logger.warning(f"Expected 16-channel latent, got {latent.shape[1]} channels. "
                         f"Make sure you're using Wan VAE (qwen_image_vae.safetensors), "
                         f"not standard 4-channel VAE!")
            debug_info.append(f"WARNING: Got {latent.shape[1]}-channel latent (expected 16)")

        debug_info.append(f"Encoded latent shape: {latent.shape}")

        # Add temporal dimension for Wan Video
        # Shape: [B, C, H, W] → [B, C, 1, H, W]
        # Single frame (T=1) for first frame conditioning
        if len(latent.shape) == 4:
            latent_temporal = latent.unsqueeze(2)  # Add T dimension at position 2
            debug_info.append(f"Added temporal dimension: {latent.shape} → {latent_temporal.shape}")
        else:
            # Already has temporal dimension
            latent_temporal = latent
            debug_info.append(f"Latent already has temporal dimension: {latent_temporal.shape}")

        # Apply Wan Video normalization
        if normalize:
            # Wan Video VAE normalization (from DiffSynth-Studio)
            mean = torch.tensor([
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
            ], device=latent_temporal.device, dtype=latent_temporal.dtype)

            std = torch.tensor([
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ], device=latent_temporal.device, dtype=latent_temporal.dtype)

            # Apply normalization: (latent - mean) * (1 / std)
            # Reshape for broadcasting: [16] → [1, 16, 1, 1, 1]
            mean = mean.view(1, 16, 1, 1, 1)
            std_inv = (1.0 / std).view(1, 16, 1, 1, 1)

            latent_normalized = (latent_temporal - mean) * std_inv
            debug_info.append(f"Applied Wan normalization (mean/std)")

            # Check for NaN/Inf after normalization
            if torch.isnan(latent_normalized).any():
                logger.warning("NaN detected in normalized latent!")
                debug_info.append("WARNING: NaN in normalized latent")
            if torch.isinf(latent_normalized).any():
                logger.warning("Inf detected in normalized latent!")
                debug_info.append("WARNING: Inf in normalized latent")

            latent_final = latent_normalized
        else:
            latent_final = latent_temporal
            debug_info.append("Normalization skipped (normalize=False)")

        # Package for ComfyUI LATENT format
        # ComfyUI expects {"samples": tensor}
        latent_dict = {"samples": latent_final}

        # Add metadata for Wan Video compatibility
        latent_dict["wan_video_first_frame"] = True
        latent_dict["channels"] = latent_final.shape[1]
        latent_dict["temporal_frames"] = latent_final.shape[2]

        debug_info.append(f"Final latent shape: {latent_final.shape}")
        debug_info.append(f"Latent range: [{latent_final.min():.4f}, {latent_final.max():.4f}]")
        debug_info.append(f"Latent mean: {latent_final.mean():.4f}, std: {latent_final.std():.4f}")

        if debug_mode:
            # Show per-channel statistics
            debug_info.append("\n=== Per-Channel Statistics ===")
            for c in range(min(latent_final.shape[1], 16)):
                channel_data = latent_final[0, c]
                debug_info.append(
                    f"Channel {c}: mean={channel_data.mean():.4f}, "
                    f"std={channel_data.std():.4f}, "
                    f"range=[{channel_data.min():.4f}, {channel_data.max():.4f}]"
                )

        debug_output = "\n".join(debug_info) if debug_mode else "Enable debug_mode for details"

        if debug_mode:
            logger.info(f"[QwenToWanBridge] {debug_output}")

        return (latent_dict, debug_output)


class QwenToWanLatentSaver:
    """
    Save first frame latent to file for use with external Wan Video tools.

    Exports the 16-channel latent in formats compatible with:
    - DiffSynth-Studio Wan Video pipeline
    - Diffusers WanImageToVideoPipeline
    - Custom inference scripts
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "First frame latent from QwenToWanFirstFrameLatent"
                }),
                "filename_prefix": ("STRING", {
                    "default": "qwen_first_frame",
                    "tooltip": "Prefix for saved latent file"
                }),
            },
            "optional": {
                "format": (["safetensors", "pt", "npz"], {
                    "default": "safetensors",
                    "tooltip": "File format (safetensors recommended)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_latent"
    CATEGORY = "QwenWanBridge"
    TITLE = "Save Qwen First Frame Latent"
    DESCRIPTION = "Export first frame latent for external Wan Video tools"
    OUTPUT_NODE = True

    def save_latent(self, latent: dict, filename_prefix: str, format: str = "safetensors"):
        """
        Save first frame latent to file.

        Args:
            latent: Latent dict from QwenToWanFirstFrameLatent
            filename_prefix: Prefix for filename
            format: File format (safetensors, pt, npz)

        Returns:
            Saved file path
        """
        import os

        # Get latent tensor
        latent_tensor = latent["samples"]

        # Create output directory
        output_dir = folder_paths.get_output_directory() if COMFY_AVAILABLE else "./output"
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}"

        if format == "safetensors":
            try:
                from safetensors.torch import save_file
                filepath = os.path.join(output_dir, f"{filename}.safetensors")

                # Prepare metadata
                metadata = {
                    "channels": str(latent_tensor.shape[1]),
                    "temporal_frames": str(latent_tensor.shape[2]),
                    "height": str(latent_tensor.shape[3]),
                    "width": str(latent_tensor.shape[4]),
                    "source": "QwenImageEdit",
                    "target": "WanVideo",
                }

                # Save
                save_file({"latent": latent_tensor}, filepath, metadata=metadata)
                logger.info(f"Saved first frame latent to {filepath}")

            except ImportError:
                raise RuntimeError("safetensors not available. Install: pip install safetensors")

        elif format == "pt":
            filepath = os.path.join(output_dir, f"{filename}.pt")
            torch.save({
                "latent": latent_tensor,
                "metadata": {
                    "channels": latent_tensor.shape[1],
                    "temporal_frames": latent_tensor.shape[2],
                    "source": "QwenImageEdit",
                    "target": "WanVideo",
                }
            }, filepath)
            logger.info(f"Saved first frame latent to {filepath}")

        elif format == "npz":
            filepath = os.path.join(output_dir, f"{filename}.npz")
            np.savez_compressed(
                filepath,
                latent=latent_tensor.cpu().numpy(),
                metadata=np.array([{
                    "channels": latent_tensor.shape[1],
                    "temporal_frames": latent_tensor.shape[2],
                    "source": "QwenImageEdit",
                    "target": "WanVideo",
                }], dtype=object)
            )
            logger.info(f"Saved first frame latent to {filepath}")

        else:
            raise ValueError(f"Unknown format: {format}")

        return (filepath,)


class QwenToWanImageSaver:
    """
    Save Qwen-edited first frame as image for visual verification.

    Useful for checking edit quality before video generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Qwen-edited image (first frame)"
                }),
                "filename_prefix": ("STRING", {
                    "default": "qwen_first_frame",
                    "tooltip": "Prefix for saved image"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_image"
    CATEGORY = "QwenWanBridge"
    TITLE = "Save Qwen First Frame Image"
    DESCRIPTION = "Save edited first frame for verification"
    OUTPUT_NODE = True

    def save_image(self, image: torch.Tensor, filename_prefix: str):
        """
        Save first frame image.

        Args:
            image: Image tensor [B, H, W, C]
            filename_prefix: Prefix for filename

        Returns:
            Saved file path
        """
        import os

        # Convert to PIL
        # ComfyUI format: [B, H, W, C] with values 0-1
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Create output directory
        output_dir = folder_paths.get_output_directory() if COMFY_AVAILABLE else "./output"
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        # Save
        pil_image.save(filepath, compress_level=4)
        logger.info(f"Saved first frame image to {filepath}")

        return (filepath,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenToWanFirstFrameLatent": QwenToWanFirstFrameLatent,
    "QwenToWanLatentSaver": QwenToWanLatentSaver,
    "QwenToWanImageSaver": QwenToWanImageSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenToWanFirstFrameLatent": "Qwen → Wan First Frame Latent",
    "QwenToWanLatentSaver": "Save First Frame Latent (Wan)",
    "QwenToWanImageSaver": "Save First Frame Image",
}
