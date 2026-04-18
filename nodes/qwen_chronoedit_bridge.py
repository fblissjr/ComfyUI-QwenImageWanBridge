"""
QwenToChronoEditBridge

Bridge node for integrating Qwen Image Edit output with ChronoEdit video generation.
Prepares Qwen-edited images for use with Kijai's WanVideoWrapper ChronoEdit nodes.

Author: ComfyUI-QwenImageWanBridge
License: Apache 2.0
"""

import torch
import torch.nn.functional as F
import comfy.model_management as mm


class QwenToChronoEditBridge:
    """
    Bridge node connecting Qwen Image Edit → ChronoEdit Video Generation

    Takes a Qwen-edited image and ensures proper formatting for ChronoEdit pipeline:
    - Enforces 32-pixel alignment (required for Wan VAE)
    - Validates frame count format (4n+1 for temporal consistency)
    - Outputs image ready for both CLIP encoding and VAE encoding

    Workflow Integration:
        QwenVLTextEncoder → VAEDecode → QwenToChronoEditBridge
                                              ↓
                    ┌─────────────────────────┴──────────────────────┐
                    ↓                                                ↓
        WanVideoClipVisionEncode                    WanVideoImageToVideoEncode
                    ↓                                                ↓
            (clip_embeds)                                    (image_embeds)
                    └─────────────────────┬──────────────────────────┘
                                          ↓
                                WanVideoSampler → Video Output
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {
                    "default": 832,
                    "min": 256,
                    "max": 2048,
                    "step": 32,
                    "tooltip": "Target width (must be 32-pixel aligned for Wan VAE)"
                }),
                "target_height": ("INT", {
                    "default": 480,
                    "min": 256,
                    "max": 2048,
                    "step": 32,
                    "tooltip": "Target height (must be 32-pixel aligned for Wan VAE)"
                }),
                "num_frames": ("INT", {
                    "default": 17,
                    "min": 5,
                    "max": 161,
                    "step": 4,
                    "tooltip": "Number of video frames to generate (must be 4n+1: 17, 81, 161)"
                }),
                "resize_mode": (["stretch", "fit", "fill"], {
                    "default": "fit",
                    "tooltip": "stretch: direct resize | fit: maintain aspect (letterbox) | fill: crop to fill"
                }),
            },
            "optional": {
                "debug_info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print debug information about image dimensions and processing"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "num_frames", "info")
    FUNCTION = "prepare_for_chronoedit"
    CATEGORY = "QwenWanBridge"
    DESCRIPTION = "Prepares Qwen-edited images for ChronoEdit video generation pipeline"

    def prepare_for_chronoedit(self, image, target_width, target_height, num_frames,
                               resize_mode="fit", debug_info=False):
        """
        Prepare Qwen-edited image for ChronoEdit pipeline

        Args:
            image: Input image tensor [B, H, W, C] from Qwen VAEDecode
            target_width: Target width (32px aligned)
            target_height: Target height (32px aligned)
            num_frames: Number of frames for video (4n+1 format)
            resize_mode: How to handle aspect ratio
            debug_info: Print processing details

        Returns:
            image: Prepared image tensor
            width: Final width (guaranteed 32px aligned)
            height: Final height (guaranteed 32px aligned)
            num_frames: Validated frame count
            info: Processing information string
        """

        device = mm.get_torch_device()
        batch_size, orig_h, orig_w, channels = image.shape

        # Enforce 32-pixel alignment for Wan VAE
        target_width = ((target_width + 31) // 32) * 32
        target_height = ((target_height + 31) // 32) * 32

        # Validate frame count (must be 4n+1)
        if (num_frames - 1) % 4 != 0:
            # Round to nearest valid frame count
            num_frames = ((num_frames - 1) // 4) * 4 + 1
            if debug_info:
                print(f"[QwenToChronoEditBridge] Adjusted frame count to {num_frames} (must be 4n+1)")

        info_lines = []
        info_lines.append(f"Input: {orig_w}×{orig_h} ({batch_size} images)")
        info_lines.append(f"Target: {target_width}×{target_height} (32px aligned)")
        info_lines.append(f"Frames: {num_frames} (4n+1 format)")
        info_lines.append(f"Mode: {resize_mode}")

        # Process image based on resize mode
        processed_image = image.clone()

        if resize_mode == "stretch":
            # Direct resize - may distort aspect ratio
            if orig_w != target_width or orig_h != target_height:
                # Convert from [B, H, W, C] to [B, C, H, W]
                processed_image = processed_image.permute(0, 3, 1, 2)
                processed_image = F.interpolate(
                    processed_image,
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                )
                # Convert back to [B, H, W, C]
                processed_image = processed_image.permute(0, 2, 3, 1)
                info_lines.append(f"Resized: stretch to {target_width}×{target_height}")

        elif resize_mode == "fit":
            # Maintain aspect ratio - letterbox if needed
            aspect_orig = orig_w / orig_h
            aspect_target = target_width / target_height

            if abs(aspect_orig - aspect_target) < 0.01:
                # Aspect ratios match - direct resize
                processed_image = processed_image.permute(0, 3, 1, 2)
                processed_image = F.interpolate(
                    processed_image,
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                )
                processed_image = processed_image.permute(0, 2, 3, 1)
                info_lines.append(f"Resized: fit (aspect ratio preserved)")
            else:
                # Letterbox to maintain aspect
                if aspect_orig > aspect_target:
                    # Image is wider - fit to width
                    scale_h = int(target_width / aspect_orig)
                    scale_h = ((scale_h + 31) // 32) * 32  # Ensure alignment
                    pad_top = (target_height - scale_h) // 2
                    pad_bottom = target_height - scale_h - pad_top

                    processed_image = processed_image.permute(0, 3, 1, 2)
                    processed_image = F.interpolate(
                        processed_image,
                        size=(scale_h, target_width),
                        mode='bilinear',
                        align_corners=False
                    )
                    processed_image = F.pad(processed_image, (0, 0, pad_top, pad_bottom), value=0)
                    processed_image = processed_image.permute(0, 2, 3, 1)
                    info_lines.append(f"Resized: letterbox (width-fit, padded {pad_top}+{pad_bottom}px)")
                else:
                    # Image is taller - fit to height
                    scale_w = int(target_height * aspect_orig)
                    scale_w = ((scale_w + 31) // 32) * 32  # Ensure alignment
                    pad_left = (target_width - scale_w) // 2
                    pad_right = target_width - scale_w - pad_left

                    processed_image = processed_image.permute(0, 3, 1, 2)
                    processed_image = F.interpolate(
                        processed_image,
                        size=(target_height, scale_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    processed_image = F.pad(processed_image, (pad_left, pad_right, 0, 0), value=0)
                    processed_image = processed_image.permute(0, 2, 3, 1)
                    info_lines.append(f"Resized: letterbox (height-fit, padded {pad_left}+{pad_right}px)")

        elif resize_mode == "fill":
            # Center crop to fill target dimensions
            aspect_orig = orig_w / orig_h
            aspect_target = target_width / target_height

            if abs(aspect_orig - aspect_target) < 0.01:
                # Aspect ratios match - direct resize
                processed_image = processed_image.permute(0, 3, 1, 2)
                processed_image = F.interpolate(
                    processed_image,
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                )
                processed_image = processed_image.permute(0, 2, 3, 1)
                info_lines.append(f"Resized: fill (aspect ratio preserved)")
            else:
                # Scale and crop to fill
                if aspect_orig > aspect_target:
                    # Image is wider - fit to height and crop width
                    scale_w = int(target_height * aspect_orig)

                    processed_image = processed_image.permute(0, 3, 1, 2)
                    processed_image = F.interpolate(
                        processed_image,
                        size=(target_height, scale_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    # Center crop width
                    crop_left = (scale_w - target_width) // 2
                    processed_image = processed_image[:, :, :, crop_left:crop_left + target_width]
                    processed_image = processed_image.permute(0, 2, 3, 1)
                    info_lines.append(f"Resized: fill (height-fit, cropped {crop_left}px from sides)")
                else:
                    # Image is taller - fit to width and crop height
                    scale_h = int(target_width / aspect_orig)

                    processed_image = processed_image.permute(0, 3, 1, 2)
                    processed_image = F.interpolate(
                        processed_image,
                        size=(scale_h, target_width),
                        mode='bilinear',
                        align_corners=False
                    )
                    # Center crop height
                    crop_top = (scale_h - target_height) // 2
                    processed_image = processed_image[:, :, crop_top:crop_top + target_height, :]
                    processed_image = processed_image.permute(0, 2, 3, 1)
                    info_lines.append(f"Resized: fill (width-fit, cropped {crop_top}px from top/bottom)")

        # Ensure proper range [0, 1] for ComfyUI
        processed_image = torch.clamp(processed_image, 0.0, 1.0)

        # Final validation
        final_h, final_w = processed_image.shape[1:3]
        assert final_w == target_width, f"Width mismatch: {final_w} != {target_width}"
        assert final_h == target_height, f"Height mismatch: {final_h} != {target_height}"
        assert final_w % 32 == 0, f"Width not 32px aligned: {final_w}"
        assert final_h % 32 == 0, f"Height not 32px aligned: {final_h}"

        info_lines.append(f"Output: {final_w}×{final_h} (validated)")
        info_text = "\n".join(info_lines)

        if debug_info:
            print(f"\n[QwenToChronoEditBridge] Processing Summary:")
            print(info_text)

        return (processed_image, target_width, target_height, num_frames, info_text)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenToChronoEditBridge": QwenToChronoEditBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenToChronoEditBridge": "Qwen to ChronoEdit Bridge",
}
