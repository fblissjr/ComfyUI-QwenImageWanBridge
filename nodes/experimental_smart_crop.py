"""
QwenSmartCrop - Experimental intelligent image cropping for face isolation
Uses multiple strategies including VLM-based detection via shrug-prompter
experimental feature testing
"""

import torch
import logging
import math
from typing import Optional, Tuple, Dict, Any
import base64
import io
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

try:
    import comfy.utils

    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    logger.warning("ComfyUI utilities not available")


class QwenSmartCrop:
    """
    Experimental smart cropping with multiple detection strategies.

    Strategies:
    1. Geometric: Simple center/portrait heuristics
    2. Saliency: Edge/variance detection
    3. VLM: Use Qwen3-VL via shrug-prompter to detect headshot location
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to crop"}),
                "detection_mode": (
                    [
                        "center_square",
                        "portrait_auto",
                        "saliency_crop",
                        "vlm_detect",
                        "auto_fallback",
                    ],
                    {
                        "default": "auto_fallback",
                        "tooltip": (
                            "Detection strategy:\n\n"
                            "center_square: Simple center crop to square\n"
                            "portrait_auto: Upper-center bias for portraits\n"
                            "saliency_crop: Edge/variance detection\n"
                            "vlm_detect: Use Qwen3-VL to find face (requires VLM_CONTEXT)\n"
                            "auto_fallback: Try VLM, fall back to saliency, then geometric"
                        ),
                    },
                ),
                "padding": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Padding around detected region (0.2 = 20% expansion)",
                    },
                ),
                "output_square": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Force output to square aspect ratio"},
                ),
                "square_anchor": (
                    ["face_headshot", "top", "center", "bottom", "left", "right"],
                    {
                        "default": "face_headshot",
                        "tooltip": (
                            "When making square crops, anchor to:\n\n"
                            "face_headshot: Uses bbox WIDTH for square size - perfect for face crops (avoids body)\n"
                            "top: Align to top of bbox, use full padded height (may include shoulders)\n"
                            "center: Center on detected bbox (may include extra area)\n"
                            "bottom: Align to bottom of bbox\n"
                            "left/right: Align horizontally"
                        ),
                    },
                ),
                "min_crop_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Minimum crop dimension in pixels",
                    },
                ),
            },
            "optional": {
                "vlm_context": (
                    "VLM_CONTEXT",
                    {
                        "tooltip": "Required for vlm_detect mode - connect from shrug-prompter"
                    },
                ),
                "vlm_prompt": (
                    "STRING",
                    {
                        "default": "Locate the primary face in this image. Report the bbox coordinates in JSON format.",
                        "multiline": True,
                        "tooltip": "Custom VLM detection prompt for Qwen3-VL (expects JSON with bbox_2d)",
                    },
                ),
                "vlm_max_tokens": (
                    "INT",
                    {
                        "default": 100,
                        "min": 10,
                        "max": 500,
                        "step": 10,
                        "tooltip": "Max tokens for VLM response (bbox should be short)",
                    },
                ),
                "vlm_temperature": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "VLM temperature (lower = more deterministic bbox)",
                    },
                ),
                "vlm_top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "VLM top_p sampling",
                    },
                ),
                "debug_mode": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Output debug visualization"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("cropped_image", "info", "debug_viz")
    FUNCTION = "smart_crop"
    CATEGORY = "QwenImage/Experimental"
    TITLE = "Qwen Smart Crop (Experimental)"
    DESCRIPTION = "Intelligent face cropping with multiple detection strategies. Experimental - not production ready!"

    def geometric_crop(
        self,
        image: torch.Tensor,
        mode: str,
        padding: float,
        output_square: bool,
        square_anchor: str,
        min_size: int,
    ) -> Tuple[torch.Tensor, str]:
        """Simple geometric cropping strategies - zero dependencies"""
        b, h, w, c = image.shape

        if mode == "center_square":
            size = min(h, w)
            y = (h - size) // 2
            x = (w - size) // 2
            cropped = image[:, y : y + size, x : x + size, :]
            info = f"Center square: {size}x{size} from ({x},{y})"

        elif mode == "portrait_auto":
            # Portrait: upper-center bias
            # Landscape: center square
            if h > w:  # Portrait orientation
                size = w
                # Start from 10% down, capture square region
                y = int(h * 0.1)
                x = 0
                crop_h = min(size, h - y)
                cropped = image[:, y : y + crop_h, x : x + size, :]

                # Pad if needed to make square
                if crop_h < size and output_square:
                    pad_bottom = size - crop_h
                    cropped = torch.nn.functional.pad(
                        cropped, (0, 0, 0, 0, 0, pad_bottom)
                    )

                info = f"Portrait auto: {size}x{crop_h} from ({x},{y})"
            else:  # Landscape
                size = min(h, w)
                y = (h - size) // 2
                x = (w - size) // 2
                cropped = image[:, y : y + size, x : x + size, :]
                info = f"Landscape center: {size}x{size} from ({x},{y})"

        else:
            raise ValueError(f"Unknown geometric mode: {mode}")

        return cropped, info

    def saliency_crop(
        self,
        image: torch.Tensor,
        padding: float,
        output_square: bool,
        square_anchor: str,
        min_size: int,
    ) -> Tuple[torch.Tensor, str]:
        """Edge/variance based saliency detection"""
        b, h, w, c = image.shape

        # Convert to grayscale
        gray = image.mean(dim=-1)  # [B, H, W]

        # Compute gradient magnitude (edges)
        # Sobel-like edge detection
        dy = torch.abs(gray[:, 1:, :] - gray[:, :-1, :])
        dx = torch.abs(gray[:, :, 1:] - gray[:, :, :-1])

        # Create edge magnitude map
        edge_map = torch.zeros_like(gray)
        edge_map[:, :-1, :] += dy
        edge_map[:, :, :-1] += dx

        # Find high-edge regions (likely faces/subjects)
        threshold = edge_map.quantile(0.7)
        mask = edge_map > threshold

        # Get bounding box of salient region
        coords = torch.nonzero(mask[0])  # [N, 2] in (y, x) format

        if len(coords) > 0:
            y_min, y_max = coords[:, 0].min().item(), coords[:, 0].max().item()
            x_min, x_max = coords[:, 1].min().item(), coords[:, 1].max().item()

            # Calculate center and size with padding
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2
            raw_h = y_max - y_min
            raw_w = x_max - x_min

            # Add padding
            padded_h = int(raw_h * (1 + padding))
            padded_w = int(raw_w * (1 + padding))

            # Make square if requested
            if output_square:
                size = max(padded_h, padded_w)
                size = max(size, min_size)
                size = min(size, w, h)  # Cap to image bounds

                # Calculate crop bounds
                y1 = max(0, center_y - size // 2)
                x1 = max(0, center_x - size // 2)
                y2 = min(h, y1 + size)
                x2 = min(w, x1 + size)

                # Adjust if we hit bounds
                if y2 - y1 < size:
                    y1 = max(0, y2 - size)
                if x2 - x1 < size:
                    x1 = max(0, x2 - size)

                cropped = image[:, y1:y2, x1:x2, :]
                info = f"Saliency square: {x2 - x1}x{y2 - y1} from ({x1},{y1}), center ({center_x},{center_y})"
            else:
                # Non-square
                y1 = max(0, center_y - padded_h // 2)
                x1 = max(0, center_x - padded_w // 2)
                y2 = min(h, y1 + padded_h)
                x2 = min(w, x1 + padded_w)

                cropped = image[:, y1:y2, x1:x2, :]
                info = f"Saliency crop: {x2 - x1}x{y2 - y1} from ({x1},{y1})"
        else:
            # Fallback to center
            logger.warning(
                "Saliency detection found no edges, falling back to center crop"
            )
            cropped, info = self.geometric_crop(
                image, "center_square", padding, output_square, min_size
            )
            info = f"Saliency fallback: {info}"

        return cropped, info

    def vlm_detect_crop(
        self,
        image: torch.Tensor,
        vlm_context: Any,
        vlm_prompt: str,
        padding: float,
        output_square: bool,
        square_anchor: str,
        min_size: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[torch.Tensor, str]:
        """Use Qwen3-VL via shrug-prompter to detect face location"""

        if vlm_context is None:
            raise ValueError(
                "VLM context required for vlm_detect mode. Connect shrug-prompter VLMProviderConfig."
            )

        b, h, w, c = image.shape

        # Convert image to JPEG bytes (same pattern as shrug-prompter VLMPrompter)
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Get raw JPEG bytes for API call
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
        img_bytes = buffered.getvalue()
        buffered.close()

        try:
            # Call VLM via shrug-prompter router
            response = self._call_vlm(
                vlm_context, img_bytes, vlm_prompt, max_tokens, temperature, top_p
            )

            # Parse response - expecting "x1,y1,x2,y2" in percentages
            coords = self._parse_bbox_response(response)

            if coords:
                x1_pct, y1_pct, x2_pct, y2_pct = coords
                logger.info(
                    f"[QwenSmartCrop] Percentage coords: x1={x1_pct}, y1={y1_pct}, x2={x2_pct}, y2={y2_pct}"
                )

                # Convert percentages to pixels
                x1 = int(w * x1_pct / 100)
                y1 = int(h * y1_pct / 100)
                x2 = int(w * x2_pct / 100)
                y2 = int(h * y2_pct / 100)
                logger.info(
                    f"[QwenSmartCrop] Pixel coords before padding: x1={x1}, y1={y1}, x2={x2}, y2={y2}"
                )
                logger.info(f"[QwenSmartCrop] Image size: {w}x{h}")

                # Add padding
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                logger.info(
                    f"[QwenSmartCrop] BBox size: {bbox_w}x{bbox_h}, center: ({center_x},{center_y})"
                )

                padded_w = int(bbox_w * (1 + padding))
                padded_h = int(bbox_h * (1 + padding))
                logger.info(f"[QwenSmartCrop] Padded size: {padded_w}x{padded_h}")

                if output_square:
                    # Apply anchor positioning
                    if square_anchor == "face_headshot":
                        # Use WIDTH (not height) for square size - perfect for headshots
                        # Faces are wider than tall vertically, so width-based square crops just the head
                        size = max(padded_w, min_size)
                        size = min(size, w, h)  # Cap to image bounds
                        logger.info(
                            f"[QwenSmartCrop] Face headshot mode: using width {padded_w}px for square size"
                        )

                        # Anchor to top of bbox
                        y1_crop = y1
                        y2_crop = min(h, y1_crop + size)
                        x1_crop = max(0, center_x - size // 2)
                        x2_crop = min(w, x1_crop + size)
                        # Adjust horizontally if hit bounds
                        if x2_crop - x1_crop < size:
                            x1_crop = max(0, x2_crop - size)

                        logger.info(
                            f"[QwenSmartCrop] Square size (face_headshot): {size}px (bbox_w={bbox_w}, padded_w={padded_w})"
                        )

                    else:
                        # All other anchors use full padded dimensions
                        size = max(padded_w, padded_h, min_size)
                        size = min(size, w, h)
                        logger.info(
                            f"[QwenSmartCrop] Square size (capped): {size} (was {max(padded_w, padded_h)})"
                        )

                        if square_anchor == "top":
                            # Anchor to top of bbox
                            y1_crop = y1
                            y2_crop = min(h, y1_crop + size)
                            x1_crop = max(0, center_x - size // 2)
                            x2_crop = min(w, x1_crop + size)
                            if x2_crop - x1_crop < size:
                                x1_crop = max(0, x2_crop - size)
                        elif square_anchor == "bottom":
                            y2_crop = y2
                            y1_crop = max(0, y2_crop - size)
                            x1_crop = max(0, center_x - size // 2)
                            x2_crop = min(w, x1_crop + size)
                            if x2_crop - x1_crop < size:
                                x1_crop = max(0, x2_crop - size)
                        elif square_anchor == "left":
                            x1_crop = x1
                            x2_crop = min(w, x1_crop + size)
                            y1_crop = max(0, center_y - size // 2)
                            y2_crop = min(h, y1_crop + size)
                            if y2_crop - y1_crop < size:
                                y1_crop = max(0, y2_crop - size)
                        elif square_anchor == "right":
                            x2_crop = x2
                            x1_crop = max(0, x2_crop - size)
                            y1_crop = max(0, center_y - size // 2)
                            y2_crop = min(h, y1_crop + size)
                            if y2_crop - y1_crop < size:
                                y1_crop = max(0, y2_crop - size)
                        else:  # center
                            x1_crop = max(0, center_x - size // 2)
                            y1_crop = max(0, center_y - size // 2)
                            x2_crop = min(w, x1_crop + size)
                            y2_crop = min(h, y1_crop + size)
                            if x2_crop - x1_crop < size:
                                x1_crop = max(0, x2_crop - size)
                            if y2_crop - y1_crop < size:
                                y1_crop = max(0, y2_crop - size)

                    logger.info(
                        f"[QwenSmartCrop] Final crop coords: x1={x1_crop}, y1={y1_crop}, x2={x2_crop}, y2={y2_crop}"
                    )
                else:
                    x1_crop = max(0, center_x - padded_w // 2)
                    y1_crop = max(0, center_y - padded_h // 2)
                    x2_crop = min(w, x1_crop + padded_w)
                    y2_crop = min(h, y1_crop + padded_h)
                    logger.info(
                        f"[QwenSmartCrop] Final crop coords (non-square): x1={x1_crop}, y1={y1_crop}, x2={x2_crop}, y2={y2_crop}"
                    )

                cropped = image[:, y1_crop:y2_crop, x1_crop:x2_crop, :]
                logger.info(f"[QwenSmartCrop] Cropped shape: {cropped.shape}")
                info = f"VLM detected: {x2_crop - x1_crop}x{y2_crop - y1_crop} from ({x1_crop},{y1_crop})"

                return cropped, info
            else:
                raise ValueError("Could not parse VLM bbox response")

        except Exception as e:
            logger.warning(f"VLM detection failed: {e}, falling back to saliency")
            cropped, info = self.saliency_crop(image, padding, output_square, min_size)
            info = f"VLM fallback: {info}"
            return cropped, info

    def _call_vlm(
        self,
        vlm_context: Any,
        image_bytes: bytes,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> str:
        """Call VLM API via shrug-prompter router (heylookitsanllm compatible)"""

        # Import shrug_router for API calls
        import sys
        import os

        shrug_path = os.path.expanduser("shrug-prompter")
        if shrug_path not in sys.path:
            sys.path.insert(0, shrug_path)

        try:
            from shrug_router import send_request
        except ImportError:
            raise ImportError("Could not import shrug_router node")

        # Handle both flat and nested context structures (from shrug-prompter pattern)
        if isinstance(vlm_context, dict) and "provider_config" in vlm_context:
            provider_config = vlm_context["provider_config"]
        else:
            provider_config = vlm_context

        base_url = provider_config.get("base_url", "")

        # Check if we should use multipart (raw bytes) for better performance
        try:
            from api.capabilities_detector import CapabilityDetector

            use_multipart = CapabilityDetector.should_use_multipart(base_url)
        except ImportError:
            # Default to multipart for heylookitsanllm
            use_multipart = "8080" in base_url or "localhost" in base_url

        # Build messages based on endpoint type
        if use_multipart:
            # Multipart endpoint - raw image bytes
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise computer vision assistant. Respond only with the requested data format.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "__RAW_IMAGE__"}},
                    ],
                },
            ]
        else:
            # Standard base64 endpoint
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise computer vision assistant. Respond only with the requested data format.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                },
            ]

        # Prepare request kwargs
        kwargs = {
            "provider": provider_config.get("provider", "openai"),
            "base_url": base_url,
            "api_key": provider_config.get("api_key", ""),
            "llm_model": provider_config.get("llm_model", "Qwen3-VL-4B-Instruct-6bit"),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": None,
        }

        # Add raw images if using multipart
        if use_multipart:
            kwargs["raw_images"] = [image_bytes]
            kwargs["image_quality"] = 85
            kwargs["preserve_alpha"] = False

        try:
            response = send_request(**kwargs)

            # Extract text from response
            if isinstance(response, dict) and "choices" in response:
                if len(response["choices"]) > 0:
                    content = response["choices"][0]["message"]["content"]
                    logger.info(f"[QwenSmartCrop] VLM response: {content}")
                    return content

            raise ValueError(f"Unexpected response format: {response}")

        except Exception as e:
            logger.error(f"VLM API call failed: {e}")
            raise

    def _parse_bbox_response(
        self, response: str
    ) -> Optional[Tuple[float, float, float, float]]:
        """Parse VLM response for bbox coordinates (Qwen3-VL format: 0-1000)"""
        import re
        import json

        logger.info(f"[QwenSmartCrop] Parsing bbox from response: '{response}'")

        # Try JSON format first (Qwen3-VL default)
        try:
            # Clean markdown fencing if present
            cleaned = response
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]

            data = json.loads(cleaned.strip())

            # Handle list or single object
            if isinstance(data, list):
                data = data[0] if len(data) > 0 else {}

            # Extract bbox_2d (Qwen3-VL format)
            if "bbox_2d" in data:
                coords = data["bbox_2d"]
                logger.info(f"[QwenSmartCrop] Found JSON bbox_2d: {coords}")

                if len(coords) >= 4:
                    # Qwen3-VL uses 0-1000 range
                    # Convert to 0-100 percentages
                    coords_pct = [c / 10.0 for c in coords[:4]]
                    logger.info(
                        f"[QwenSmartCrop] Converted 0-1000 → 0-100: {coords_pct}"
                    )

                    if all(0 <= c <= 100 for c in coords_pct):
                        return tuple(coords_pct)

        except json.JSONDecodeError:
            logger.info(f"[QwenSmartCrop] Not JSON format, trying numeric parsing")
        except Exception as e:
            logger.warning(f"[QwenSmartCrop] JSON parsing error: {e}")

        # Fallback: Try to find raw numbers
        numbers = re.findall(r"\d+(?:\.\d+)?", response)
        logger.info(f"[QwenSmartCrop] Extracted numbers: {numbers}")

        if len(numbers) >= 4:
            try:
                coords = [float(n) for n in numbers[:4]]
                logger.info(f"[QwenSmartCrop] Parsed coords: {coords}")

                # Check if 0-1000 range (Qwen3-VL)
                if all(0 <= c <= 1000 for c in coords):
                    coords_pct = [c / 10.0 for c in coords]
                    logger.info(
                        f"[QwenSmartCrop] Converted 0-1000 → 0-100: {coords_pct}"
                    )
                    return tuple(coords_pct)

                # Check if 0-100 range (old format)
                elif all(0 <= c <= 100 for c in coords):
                    logger.info(
                        f"[QwenSmartCrop] Valid bbox found (0-100 range): {coords}"
                    )
                    return tuple(coords)

                else:
                    logger.warning(
                        f"[QwenSmartCrop] Coords out of expected range: {coords}"
                    )

            except ValueError as e:
                logger.warning(f"[QwenSmartCrop] ValueError parsing coords: {e}")

        logger.warning(
            f"[QwenSmartCrop] Could not parse valid bbox from VLM response: '{response}'"
        )
        return None

    def create_debug_viz(
        self, original: torch.Tensor, cropped: torch.Tensor, crop_info: str
    ) -> torch.Tensor:
        """Create debug visualization showing crop region"""
        # For now, just return the cropped image
        # Could enhance with bbox overlay on original
        return cropped

    def smart_crop(
        self,
        image: torch.Tensor,
        detection_mode: str,
        padding: float = 0.2,
        output_square: bool = True,
        square_anchor: str = "top",
        min_crop_size: int = 256,
        vlm_context: Optional[Any] = None,
        vlm_prompt: str = "",
        vlm_max_tokens: int = 100,
        vlm_temperature: float = 0.1,
        vlm_top_p: float = 0.95,
        debug_mode: bool = False,
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:
        """
        Main smart crop function with multiple strategies
        """

        if not COMFY_AVAILABLE:
            raise RuntimeError("ComfyUI utilities not available")

        info_lines = [f"Detection mode: {detection_mode}"]

        try:
            if detection_mode == "center_square":
                cropped, crop_info = self.geometric_crop(
                    image,
                    "center_square",
                    padding,
                    output_square,
                    square_anchor,
                    min_crop_size,
                )

            elif detection_mode == "portrait_auto":
                cropped, crop_info = self.geometric_crop(
                    image,
                    "portrait_auto",
                    padding,
                    output_square,
                    square_anchor,
                    min_crop_size,
                )

            elif detection_mode == "saliency_crop":
                cropped, crop_info = self.saliency_crop(
                    image, padding, output_square, square_anchor, min_crop_size
                )

            elif detection_mode == "vlm_detect":
                if not vlm_prompt:
                    vlm_prompt = "Locate the primary face in this image. Report the bbox coordinates in JSON format."

                cropped, crop_info = self.vlm_detect_crop(
                    image,
                    vlm_context,
                    vlm_prompt,
                    padding,
                    output_square,
                    square_anchor,
                    min_crop_size,
                    vlm_max_tokens,
                    vlm_temperature,
                    vlm_top_p,
                )

            elif detection_mode == "auto_fallback":
                # Try VLM first if context available
                if vlm_context is not None:
                    try:
                        if not vlm_prompt:
                            vlm_prompt = "Locate the primary face in this image. Report the bbox coordinates in JSON format."

                        cropped, crop_info = self.vlm_detect_crop(
                            image,
                            vlm_context,
                            vlm_prompt,
                            padding,
                            output_square,
                            square_anchor,
                            min_crop_size,
                            vlm_max_tokens,
                            vlm_temperature,
                            vlm_top_p,
                        )
                        info_lines.append("Strategy: VLM detection (success)")
                    except Exception as e:
                        logger.info(f"VLM detection failed, trying saliency: {e}")
                        cropped, crop_info = self.saliency_crop(
                            image, padding, output_square, square_anchor, min_crop_size
                        )
                        info_lines.append("Strategy: Saliency (VLM failed)")
                else:
                    # No VLM, try saliency
                    cropped, crop_info = self.saliency_crop(
                        image, padding, output_square, square_anchor, min_crop_size
                    )
                    info_lines.append("Strategy: Saliency (no VLM context)")

            else:
                raise ValueError(f"Unknown detection mode: {detection_mode}")

            info_lines.append(crop_info)
            info_lines.append(f"Input: {image.shape[2]}x{image.shape[1]}")
            info_lines.append(f"Output: {cropped.shape[2]}x{cropped.shape[1]}")
            info_lines.append(f"Padding: {padding * 100:.0f}%")

            info_str = "\n".join(info_lines)

            # Debug visualization
            debug_viz = (
                self.create_debug_viz(image, cropped, crop_info)
                if debug_mode
                else cropped
            )

            logger.info(f"[QwenSmartCrop] {info_str}")

            return (cropped, info_str, debug_viz)

        except Exception as e:
            logger.error(f"Smart crop failed: {e}")
            # Ultimate fallback: return center square
            cropped, crop_info = self.geometric_crop(
                image,
                "center_square",
                padding,
                output_square,
                square_anchor,
                min_crop_size,
            )
            info_str = f"ERROR FALLBACK:\n{str(e)}\n{crop_info}"
            return (cropped, info_str, cropped)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenSmartCrop": QwenSmartCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenSmartCrop": "Qwen Smart Crop (Experimental)",
}
