"""
EliGen Entity-Level Image Generation for Qwen
Based on DiffSynth-Studio implementation
Enables precise spatial control with masks and per-region prompts
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class QwenEliGenEntityControl:
    """
    Entity-Level Image Generation (EliGen) control for Qwen models
    Allows precise spatial control using masks and entity-specific prompts

    This is a unique feature from DiffSynth-Studio not available in native ComfyUI
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "global_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape",
                    "tooltip": "Global prompt that applies to the entire image"
                }),
            },
            "optional": {
                # Entity 1
                "entity1_mask": ("MASK", {
                    "tooltip": "Mask for entity 1 (white=entity, black=ignore)"
                }),
                "entity1_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Prompt for what to generate in entity 1 region"
                }),
                "entity1_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Weight for entity 1 influence"
                }),

                # Entity 2
                "entity2_mask": ("MASK", {
                    "tooltip": "Mask for entity 2"
                }),
                "entity2_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Prompt for entity 2"
                }),
                "entity2_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),

                # Entity 3
                "entity3_mask": ("MASK", {
                    "tooltip": "Mask for entity 3"
                }),
                "entity3_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Prompt for entity 3"
                }),
                "entity3_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),

                # Entity 4
                "entity4_mask": ("MASK", {
                    "tooltip": "Mask for entity 4"
                }),
                "entity4_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Prompt for entity 4"
                }),
                "entity4_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),

                # Control options
                "apply_to_negative": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also apply entity control to negative prompt"
                }),
                "blend_mode": (["multiply", "overlay", "additive"], {
                    "default": "multiply",
                    "tooltip": "How to blend overlapping entity regions"
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show debug info about entity processing"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE")
    RETURN_NAMES = ("conditioning", "entity_preview")
    FUNCTION = "apply_entity_control"
    CATEGORY = "QwenImage/EliGen"
    TITLE = "Qwen EliGen Entity Control"
    DESCRIPTION = "Precise spatial control with masks and per-region prompts (DiffSynth feature)"

    def apply_entity_control(self, conditioning, global_prompt,
                            entity1_mask=None, entity1_prompt="", entity1_weight=1.0,
                            entity2_mask=None, entity2_prompt="", entity2_weight=1.0,
                            entity3_mask=None, entity3_prompt="", entity3_weight=1.0,
                            entity4_mask=None, entity4_prompt="", entity4_weight=1.0,
                            apply_to_negative=False, blend_mode="multiply", debug_mode=False):
        """
        Apply entity-level control to conditioning
        """

        # Collect active entities
        entities = []
        entity_masks = []

        for i, (mask, prompt, weight) in enumerate([
            (entity1_mask, entity1_prompt, entity1_weight),
            (entity2_mask, entity2_prompt, entity2_weight),
            (entity3_mask, entity3_prompt, entity3_weight),
            (entity4_mask, entity4_prompt, entity4_weight)
        ], 1):
            if mask is not None and prompt:
                entities.append({
                    "id": i,
                    "mask": mask,
                    "prompt": prompt,
                    "weight": weight
                })
                entity_masks.append(mask)

                if debug_mode:
                    logger.info(f"[EliGen] Entity {i}: prompt='{prompt[:50]}...', weight={weight}")

        if not entities:
            if debug_mode:
                logger.info("[EliGen] No entities defined, returning original conditioning")
            # Create empty preview
            preview = torch.zeros((1, 512, 512, 3))
            return (conditioning, preview)

        # Process masks and create attention masks
        attention_masks = self._create_attention_masks(entities, blend_mode, debug_mode)

        # Create entity prompts list
        entity_prompts = [e["prompt"] for e in entities]
        entity_weights = [e["weight"] for e in entities]

        # Create preview visualization
        preview = self._create_preview(entities, blend_mode)

        # Add EliGen data to conditioning
        eligen_data = {
            "eligen_entity_prompts": entity_prompts,
            "eligen_entity_masks": attention_masks,
            "eligen_entity_weights": entity_weights,
            "eligen_global_prompt": global_prompt,
            "eligen_enable_on_negative": apply_to_negative,
            "eligen_blend_mode": blend_mode
        }

        # Import node_helpers for conditioning update
        try:
            import node_helpers
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                eligen_data,
                append=True
            )

            if debug_mode:
                logger.info(f"[EliGen] Added {len(entities)} entities to conditioning")
                logger.info(f"[EliGen] Global prompt: '{global_prompt[:50]}...'")
                logger.info(f"[EliGen] Apply to negative: {apply_to_negative}")

        except ImportError:
            logger.warning("[EliGen] Could not import node_helpers, returning original conditioning")

        return (conditioning, preview)

    def _create_attention_masks(self, entities: List[Dict], blend_mode: str, debug_mode: bool) -> List[torch.Tensor]:
        """
        Create attention masks for each entity
        """
        masks = []

        for entity in entities:
            mask = entity["mask"]
            weight = entity["weight"]

            # Ensure mask is in correct format [H, W]
            if len(mask.shape) == 3:
                mask = mask.squeeze(0)
            elif len(mask.shape) == 4:
                mask = mask[0, 0]

            # Apply weight
            weighted_mask = mask * weight

            # Normalize to 0-1 range
            weighted_mask = torch.clamp(weighted_mask, 0, 1)

            masks.append(weighted_mask)

            if debug_mode:
                logger.info(f"[EliGen] Entity {entity['id']} mask shape: {mask.shape}, "
                          f"min={mask.min():.3f}, max={mask.max():.3f}")

        # Handle overlapping regions based on blend mode
        if len(masks) > 1 and blend_mode != "multiply":
            masks = self._blend_masks(masks, blend_mode, debug_mode)

        return masks

    def _blend_masks(self, masks: List[torch.Tensor], blend_mode: str, debug_mode: bool) -> List[torch.Tensor]:
        """
        Blend overlapping mask regions
        """
        if blend_mode == "overlay":
            # Normalize overlapping regions
            combined = torch.stack(masks).sum(dim=0)
            combined = torch.clamp(combined, 0, 1)

            # Redistribute weights proportionally
            for i in range(len(masks)):
                masks[i] = masks[i] / (combined + 1e-8)

        elif blend_mode == "additive":
            # Allow regions to add up (can exceed 1.0)
            pass

        if debug_mode:
            logger.info(f"[EliGen] Blended {len(masks)} masks using {blend_mode} mode")

        return masks

    def _create_preview(self, entities: List[Dict], blend_mode: str) -> torch.Tensor:
        """
        Create a preview visualization of entity masks
        """
        # Create RGB preview with different colors for each entity
        colors = [
            [1.0, 0.3, 0.3],  # Red
            [0.3, 1.0, 0.3],  # Green
            [0.3, 0.3, 1.0],  # Blue
            [1.0, 1.0, 0.3],  # Yellow
        ]

        # Get mask dimensions from first entity
        first_mask = entities[0]["mask"]
        if len(first_mask.shape) == 3:
            h, w = first_mask.shape[1:]
        else:
            h, w = first_mask.shape

        # Create preview tensor
        preview = torch.zeros((h, w, 3))

        for entity, color in zip(entities, colors):
            mask = entity["mask"]

            # Ensure mask is 2D
            if len(mask.shape) == 3:
                mask = mask.squeeze(0)
            elif len(mask.shape) == 4:
                mask = mask[0, 0]

            # Apply color
            for c in range(3):
                if blend_mode == "multiply":
                    preview[:, :, c] = torch.maximum(preview[:, :, c], mask * color[c])
                else:
                    preview[:, :, c] += mask * color[c] * 0.5

        # Normalize and add batch dimension
        preview = torch.clamp(preview, 0, 1)
        preview = preview.unsqueeze(0)

        return preview


class QwenEliGenMaskPainter:
    """
    Simple mask painter for creating entity masks
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8
                }),
                "shape": (["rectangle", "ellipse", "custom"], {
                    "default": "rectangle"
                }),
                "x": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "X position of shape center"
                }),
                "y": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Y position of shape center"
                }),
                "size_x": ("INT", {
                    "default": 256,
                    "min": 8,
                    "max": 4096,
                    "tooltip": "Width of shape"
                }),
                "size_y": ("INT", {
                    "default": 256,
                    "min": 8,
                    "max": 4096,
                    "tooltip": "Height of shape"
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Edge softness in pixels"
                }),
            },
            "optional": {
                "input_mask": ("MASK", {
                    "tooltip": "Optional input mask to add to"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask"
    CATEGORY = "QwenImage/EliGen"
    TITLE = "EliGen Mask Painter"
    DESCRIPTION = "Create simple masks for entity regions"

    def create_mask(self, width, height, shape, x, y, size_x, size_y, feather, input_mask=None):
        """
        Create or modify a mask
        """

        # Start with input mask or zeros
        if input_mask is not None:
            mask = input_mask.clone()
            # Resize if needed
            if mask.shape[-2:] != (height, width):
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode='bilinear'
                ).squeeze()
        else:
            mask = torch.zeros((height, width))

        # Create shape mask
        y_coords = torch.arange(height).unsqueeze(1).float()
        x_coords = torch.arange(width).unsqueeze(0).float()

        if shape == "rectangle":
            # Rectangle mask
            x_mask = (x_coords >= (x - size_x/2)) & (x_coords <= (x + size_x/2))
            y_mask = (y_coords >= (y - size_y/2)) & (y_coords <= (y + size_y/2))
            shape_mask = (x_mask & y_mask).float()

        elif shape == "ellipse":
            # Ellipse mask
            x_dist = ((x_coords - x) / (size_x/2)) ** 2
            y_dist = ((y_coords - y) / (size_y/2)) ** 2
            shape_mask = (x_dist + y_dist <= 1).float()

        else:  # custom
            # Just use a centered rectangle for now
            x_mask = (x_coords >= (x - size_x/2)) & (x_coords <= (x + size_x/2))
            y_mask = (y_coords >= (y - size_y/2)) & (y_coords <= (y + size_y/2))
            shape_mask = (x_mask & y_mask).float()

        # Apply feathering if requested
        if feather > 0:
            import torch.nn.functional as F
            # Use gaussian blur for feathering
            shape_mask = shape_mask.unsqueeze(0).unsqueeze(0)

            # Create gaussian kernel
            kernel_size = feather * 2 + 1
            sigma = feather / 3.0

            # Apply blur
            shape_mask = F.avg_pool2d(
                F.pad(shape_mask, (feather, feather, feather, feather), mode='reflect'),
                kernel_size=kernel_size,
                stride=1
            )
            shape_mask = shape_mask.squeeze()

        # Combine with existing mask
        mask = torch.maximum(mask, shape_mask)
        mask = torch.clamp(mask, 0, 1)

        # Add batch dimension
        mask = mask.unsqueeze(0)

        return (mask,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenEliGenEntityControl": QwenEliGenEntityControl,
    "QwenEliGenMaskPainter": QwenEliGenMaskPainter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenEliGenEntityControl": "Qwen EliGen Entity Control",
    "QwenEliGenMaskPainter": "EliGen Mask Painter",
}