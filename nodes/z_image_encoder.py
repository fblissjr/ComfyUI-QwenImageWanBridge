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
    """Load Z-Image templates from nodes/templates/z_image/*.md files."""
    templates = {}
    templates_dir = os.path.join(os.path.dirname(__file__), "templates", "z_image")

    if not os.path.exists(templates_dir):
        return templates

    for filename in os.listdir(templates_dir):
        if filename.endswith('.md'):
            template_name = filename[:-3]  # Remove '.md' suffix
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
                    "tooltip": "Add <think></think> block (auto-enabled if thinking_content provided)"
                }),
                "thinking_content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Content inside <think>...</think> tags (auto-enables think block)"
                }),
                "assistant_content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Content AFTER </think> tags (what assistant says after thinking)"
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
        thinking_content: str = "",
        assistant_content: str = "",
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    ) -> Tuple[Any, str]:

        # RAW MODE: complete control
        if raw_prompt.strip():
            formatted_text = raw_prompt
        else:
            # Determine system prompt: use provided, or fallback to template file
            effective_system = system_prompt.strip()
            if not effective_system and template_preset and template_preset != "none":
                # JS didn't fill system_prompt - load from template file as fallback
                templates = get_templates()
                if template_preset in templates:
                    effective_system = templates[template_preset]
                    logger.debug(f"Loaded template '{template_preset}' from file (JS fallback)")

            # Auto-enable think block if thinking_content provided
            use_think_block = add_think_block or bool(thinking_content.strip())
            # Build formatted prompt with chat template
            formatted_text = self._format_prompt(text, effective_system, use_think_block, thinking_content, assistant_content)

        # Encode
        tokens = clip.tokenize(formatted_text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        return (conditioning, formatted_text)

    def _format_prompt(self, text: str, system_prompt: str = "", add_think_block: bool = False, thinking_content: str = "", assistant_content: str = "") -> str:
        """
        Format prompt following Qwen3-4B chat template from tokenizer_config.json.

        Structure:
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {text}<|im_end|>
        <|im_start|>assistant
        <think>
        {thinking_content}
        </think>

        {assistant_content}
        """
        parts = []

        if system_prompt.strip():
            parts.append(f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>")

        parts.append(f"<|im_start|>user\n{text}<|im_end|>")

        # Build assistant section - match Qwen3 template exactly
        if add_think_block:
            # Format: <|im_start|>assistant\n<think>\n{content}\n</think>\n\n{assistant}
            think_inner = thinking_content.strip() if thinking_content.strip() else ""
            assistant_inner = assistant_content.strip() if assistant_content.strip() else ""
            parts.append(f"<|im_start|>assistant\n<think>\n{think_inner}\n</think>\n\n{assistant_inner}")
        else:
            # No think block: <|im_start|>assistant\n{content}
            assistant_inner = assistant_content.strip() if assistant_content.strip() else ""
            parts.append(f"<|im_start|>assistant\n{assistant_inner}")

        return "\n".join(parts)


class ZImageTextEncoderSimple:
    """
    Simple Z-Image encoder with raw output visibility.
    Follows Qwen3-4B chat template format.
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
                    "tooltip": "Add <think></think> block (auto-enabled if thinking_content provided)"
                }),
                "thinking_content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Content inside <think>...</think> tags"
                }),
                "assistant_content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Content AFTER </think> (what assistant says after thinking)"
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
        add_think_block: bool = False,
        thinking_content: str = "",
        assistant_content: str = ""
    ) -> Tuple[Any, str]:

        if raw_prompt.strip():
            formatted_text = raw_prompt
        else:
            # Auto-enable think block if thinking_content provided
            use_think_block = add_think_block or bool(thinking_content.strip())

            if use_think_block:
                # Qwen3 format: <|im_start|>assistant\n<think>\n{think}\n</think>\n\n{assistant}
                think_inner = thinking_content.strip() if thinking_content.strip() else ""
                assistant_inner = assistant_content.strip() if assistant_content.strip() else ""
                formatted_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n{think_inner}\n</think>\n\n{assistant_inner}"
            else:
                assistant_inner = assistant_content.strip() if assistant_content.strip() else ""
                formatted_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{assistant_inner}"

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
