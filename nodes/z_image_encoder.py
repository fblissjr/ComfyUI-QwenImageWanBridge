"""
Z-Image Text Encoder for ComfyUI

Transparent encoder - see exactly what gets sent to the model.
Templates auto-fill the system prompt field so you can edit them.
Use raw_prompt for complete control with your own special tokens.
"""

import os
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEQUENCE_LENGTH = 512

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_z_image_templates() -> Dict[str, str]:
    """Load Z-Image templates from nodes/templates/z_image_*.md files."""
    templates = {}
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")

    if not os.path.exists(templates_dir):
        return templates

    for filename in os.listdir(templates_dir):
        if filename.startswith('z_image_') and filename.endswith('.md'):
            template_name = filename[8:-3]  # Remove 'z_image_' prefix and '.md' suffix
            template_path = os.path.join(templates_dir, filename)
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        templates[template_name] = parts[2].strip()
            except Exception as e:
                logger.warning(f"Failed to load template {filename}: {e}")

    return templates


# Global template cache
_TEMPLATE_CACHE = None

def get_templates() -> Dict[str, str]:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = load_z_image_templates()
    return _TEMPLATE_CACHE


class ZImageTextEncoder:
    """
    Z-Image Text Encoder - transparent and editable.

    - template_preset: Dropdown loads template into system_prompt field (via JS)
    - system_prompt: Editable - modify any template or write your own
    - raw_prompt: Complete control - bypass all formatting, use your own special tokens
    - formatted_prompt output: See exactly what gets encoded
    """

    @classmethod
    def INPUT_TYPES(cls):
        templates = get_templates()
        template_names = ["none"] + sorted(templates.keys())

        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your prompt"
                }),
            },
            "optional": {
                "template_preset": (template_names, {
                    "default": "none",
                    "tooltip": "Select template - auto-fills system_prompt field (editable)"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System prompt - auto-filled by template, edit freely"
                }),
                "raw_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "RAW MODE: Bypass all formatting. Write your own <|im_start|> tokens. Ignores text/system if set."
                }),
                "add_think_block": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add <think></think> block (experimental)"
                }),
                "max_sequence_length": ("INT", {
                    "default": DEFAULT_MAX_SEQUENCE_LENGTH,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Max tokens (512 matches diffusers)"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "formatted_prompt")
    FUNCTION = "encode"
    CATEGORY = "ZImage/Encoding"
    TITLE = "Z-Image Text Encoder"

    def encode(
        self,
        clip,
        text: str,
        template_preset: str = "none",
        system_prompt: str = "",
        raw_prompt: str = "",
        add_think_block: bool = False,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    ) -> Tuple[Any, str]:

        # RAW MODE: complete control
        if raw_prompt.strip():
            formatted_text = raw_prompt
        else:
            # Build formatted prompt with chat template
            formatted_text = self._format_prompt(text, system_prompt, add_think_block)

        # Encode
        tokens = clip.tokenize(formatted_text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        return (conditioning, formatted_text)

    def _format_prompt(self, text: str, system_prompt: str = "", add_think_block: bool = False) -> str:
        parts = []

        if system_prompt.strip():
            parts.append(f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>")

        parts.append(f"<|im_start|>user\n{text}<|im_end|>")

        if add_think_block:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        else:
            parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)


class ZImageTextEncoderSimple:
    """
    Simple Z-Image encoder with raw output visibility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": True,
                }),
            },
            "optional": {
                "raw_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "RAW: Bypass formatting, use your own special tokens"
                }),
                "add_think_block": ("BOOLEAN", {
                    "default": False,
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "formatted_prompt")
    FUNCTION = "encode"
    CATEGORY = "ZImage/Encoding"
    TITLE = "Z-Image Text Encode (Simple)"

    def encode(
        self,
        clip,
        text: str,
        raw_prompt: str = "",
        add_think_block: bool = False
    ) -> Tuple[Any, str]:

        if raw_prompt.strip():
            formatted_text = raw_prompt
        elif add_think_block:
            formatted_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        else:
            formatted_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

        tokens = clip.tokenize(formatted_text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        return (conditioning, formatted_text)


# Export templates for JS
def get_z_image_template_systems() -> Dict[str, str]:
    """Get template systems for JS auto-fill. Called by __init__.py for API."""
    return get_templates()


NODE_CLASS_MAPPINGS = {
    "ZImageTextEncoder": ZImageTextEncoder,
    "ZImageTextEncoderSimple": ZImageTextEncoderSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTextEncoder": "Z-Image Text Encoder",
    "ZImageTextEncoderSimple": "Z-Image Text Encode (Simple)",
}
