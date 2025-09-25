"""
Qwen Processor V2 - Simplified for DRY principle
Only handles template formatting and drop indices
System prompts come from Template Builder
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QwenProcessorV2:
    """
    Minimal processor that only:
    1. Formats templates with vision tokens
    2. Tracks drop indices for each mode

    System prompts come from Template Builder (single source of truth)
    """

    def __init__(self):
        """Initialize with drop indices for each mode."""
        # Drop indices from DiffSynth-Studio/Diffusers
        self.drop_indices = {
            "text_to_image": 34,
            "image_edit": 64
        }

    def format_template(self,
                       text: str,
                       system_prompt: str,
                       vision_tokens: str = "") -> str:
        """
        Format the full template with system, user, and assistant sections.

        Args:
            text: User's prompt
            system_prompt: System prompt from Template Builder
            vision_tokens: Pre-formatted vision token string

        Returns:
            Formatted template string
        """
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{vision_tokens}{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def get_vision_tokens(self, num_images: int) -> str:
        """
        Generate vision token string for images.

        Args:
            num_images: Number of images

        Returns:
            Vision token string
        """
        if num_images == 0:
            return ""
        elif num_images == 1:
            return "<|vision_start|><|image_pad|><|vision_end|>"
        else:
            # Multi-image format
            parts = []
            for i in range(num_images):
                parts.append(f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>")
            return "".join(parts)

    def get_drop_index(self, mode: str) -> int:
        """
        Get the number of embeddings to drop for a given mode.

        Args:
            mode: "text_to_image" or "image_edit"

        Returns:
            Number of embeddings to drop
        """
        return self.drop_indices.get(mode, 0)