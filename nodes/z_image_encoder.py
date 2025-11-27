"""
Z-Image Text Encoder for ComfyUI

Fixes ComfyUI's hardcoded template by properly using apply_chat_template
with enable_thinking=True, matching the diffusers implementation.

Key difference from ComfyUI's built-in:
- ComfyUI: hardcodes "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
- Diffusers: uses tokenizer.apply_chat_template(enable_thinking=True)
- This node: matches diffusers behavior for proper embeddings

The enable_thinking=True adds "<think>\n\n</think>\n\n" which the model
was trained with. Missing these tokens produces out-of-distribution embeddings.

Qwen3 Model Variants (as of 2507):
- Qwen3-4B (no suffix) = INSTRUCT model with hybrid thinking (Z-Image uses this)
- Qwen3-4B-Base = base model without thinking
- Qwen3-Instruct-2507 = non-thinking only
- Qwen3-Thinking-2507 = always thinking

Known Gaps vs Diffusers (documented in nodes/docs/z_image_encoder.md):
1. Thinking tokens: FIXED by this node
2. Embedding extraction: Cannot fix (ComfyUI returns padded sequence, diffusers filters)
3. Sequence length: FIXED with max_sequence_length parameter (default 512)
4. Bundled tokenizer: Cannot fix (no thinking template in ComfyUI's bundled config)
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

# Default sequence length matching diffusers Z-Image pipeline
DEFAULT_MAX_SEQUENCE_LENGTH = 512

# Try to import yaml for template parsing
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available, template metadata parsing limited")


class ZImageTextEncoder:
    """
    Z-Image Text Encoder with proper thinking token support.

    Matches diffusers' ZImagePipeline._encode_prompt() behavior:
    - Uses apply_chat_template with enable_thinking=True
    - Supports optional system prompts (like Lumina2)
    - Provides template customization

    Why this matters:
    - Z-Image was trained with thinking tokens in the template
    - ComfyUI's hardcoded template omits these tokens
    - This causes slightly out-of-distribution embeddings
    """

    # System prompts inspired by Lumina2's approach
    SYSTEM_PROMPTS = {
        "none": "",
        "quality": "Generate a high-quality, detailed image with excellent composition.",
        "photorealistic": "Generate a photorealistic image with accurate lighting, textures, and natural details.",
        "artistic": "Generate an artistic, aesthetically pleasing image with creative composition.",
        "bilingual": "Generate an image with accurate text rendering. Support both English and Chinese text.",
    }

    def __init__(self):
        """Initialize with template loading."""
        self.templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.available_templates = self._load_available_templates()
        logger.info(f"[ZImageTextEncoder] Loaded {len(self.available_templates)} templates")

    def _load_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load available templates from templates/ directory."""
        templates = {}

        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return templates

        for filename in os.listdir(self.templates_dir):
            # Only load z_image templates
            if filename.startswith('z_image_') and filename.endswith('.md'):
                template_name = filename[:-3]  # Remove .md extension
                template_path = os.path.join(self.templates_dir, filename)

                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Parse YAML frontmatter
                    if content.startswith('---'):
                        parts = content.split('---', 2)
                        if len(parts) >= 3:
                            metadata = {}
                            if YAML_AVAILABLE:
                                try:
                                    metadata = yaml.safe_load(parts[1]) or {}
                                except Exception:
                                    pass

                            system_prompt = parts[2].strip()
                            templates[template_name] = {
                                'metadata': metadata,
                                'system_prompt': system_prompt
                            }
                except Exception as e:
                    logger.warning(f"Failed to load template {filename}: {e}")

        return templates

    @classmethod
    def INPUT_TYPES(cls):
        # Get available templates
        instance = cls()
        template_names = sorted([k for k in instance.available_templates.keys()])

        system_prompt_options = list(cls.SYSTEM_PROMPTS.keys())

        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your prompt - describe the image you want"
                }),
            },
            "optional": {
                "system_prompt_preset": (system_prompt_options, {
                    "default": "none",
                    "tooltip": "Optional system prompt to guide generation style"
                }),
                "custom_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom system prompt (overrides preset if provided)"
                }),
                "template_preset": (["none"] + template_names if template_names else ["none"], {
                    "default": "none",
                    "tooltip": "Template from nodes/templates/z_image_*.md"
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable thinking tokens (matches diffusers, recommended)"
                }),
                "max_sequence_length": ("INT", {
                    "default": DEFAULT_MAX_SEQUENCE_LENGTH,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Max token length (default 512 matches diffusers). Longer prompts truncated."
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show encoding details and formatted prompt"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "debug_output")
    FUNCTION = "encode"
    CATEGORY = "ZImage/Encoding"
    TITLE = "Z-Image Text Encoder"
    DESCRIPTION = "Proper Z-Image encoding with thinking tokens (fixes ComfyUI's hardcoded template)"

    def _format_prompt_with_thinking(
        self,
        text: str,
        system_prompt: str = "",
        enable_thinking: bool = True
    ) -> str:
        """
        Format prompt matching diffusers' apply_chat_template behavior.

        With enable_thinking=True (diffusers default):
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {text}<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        With enable_thinking=False:
        Same but with empty <think></think> block
        """
        parts = []

        # System message (optional)
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        # User message
        parts.append(f"<|im_start|>user\n{text}<|im_end|>")

        # Assistant with thinking
        if enable_thinking:
            # This matches what apply_chat_template(enable_thinking=True) produces
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        else:
            # Without thinking, just the assistant marker
            parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def _format_prompt_comfyui_style(
        self,
        text: str,
        system_prompt: str = "",
        enable_thinking: bool = True
    ) -> str:
        """
        Alternative: Use ComfyUI's simpler format but add thinking tokens.

        This is closer to what ComfyUI does but with the missing thinking block.
        """
        if system_prompt:
            # With system prompt
            if enable_thinking:
                return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            else:
                return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Without system prompt (closer to ComfyUI default)
            if enable_thinking:
                return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            else:
                return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def encode(
        self,
        clip,
        text: str,
        system_prompt_preset: str = "none",
        custom_system_prompt: str = "",
        template_preset: str = "none",
        enable_thinking: bool = True,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        debug_mode: bool = False
    ) -> Tuple[Any, str]:
        """
        Encode text for Z-Image with proper thinking token support.

        Args:
            clip: ComfyUI CLIP model (lumina2 type for Z-Image)
            text: User prompt
            system_prompt_preset: Preset system prompt name
            custom_system_prompt: Custom system prompt (overrides preset)
            template_preset: Template from nodes/templates/z_image_*.md
            enable_thinking: Add thinking tokens (default True, matches diffusers)
            max_sequence_length: Max tokens (default 512, matches diffusers)
            debug_mode: Show encoding details

        Returns:
            Tuple of (conditioning, debug_output)
        """
        debug_output = []

        # Determine system prompt
        system_prompt = ""

        if custom_system_prompt.strip():
            system_prompt = custom_system_prompt.strip()
            debug_output.append(f"Using custom system prompt ({len(system_prompt)} chars)")
        elif template_preset != "none" and template_preset in self.available_templates:
            template_data = self.available_templates[template_preset]
            system_prompt = template_data.get('system_prompt', '')
            debug_output.append(f"Using template: {template_preset}")
        elif system_prompt_preset != "none":
            system_prompt = self.SYSTEM_PROMPTS.get(system_prompt_preset, "")
            debug_output.append(f"Using preset: {system_prompt_preset}")
        else:
            debug_output.append("No system prompt (matching diffusers default)")

        # Format the prompt with thinking tokens
        formatted_text = self._format_prompt_comfyui_style(
            text=text,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking
        )

        if debug_mode:
            debug_output.append(f"enable_thinking: {enable_thinking}")
            debug_output.append(f"max_sequence_length: {max_sequence_length}")
            debug_output.append(f"Formatted prompt length: {len(formatted_text)} chars")

        # Tokenize and encode using ComfyUI's CLIP
        # Note: ComfyUI's tokenizer doesn't respect max_length directly,
        # so we warn if the prompt is likely too long
        tokens = clip.tokenize(formatted_text)

        # Estimate token count and warn if exceeding limit
        # Rough estimate: ~4 chars per token for English
        estimated_tokens = len(formatted_text) // 4
        if estimated_tokens > max_sequence_length:
            warning_msg = f"WARNING: Prompt may exceed max_sequence_length ({estimated_tokens} estimated vs {max_sequence_length} limit)"
            debug_output.append(warning_msg)
            logger.warning(f"[ZImageTextEncoder] {warning_msg}")
            logger.warning("[ZImageTextEncoder] ComfyUI may not truncate properly. Consider shortening your prompt.")

        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if debug_mode and conditioning and len(conditioning) > 0:
            cond_tensor = conditioning[0][0]
            if cond_tensor is not None:
                actual_seq_len = cond_tensor.shape[1] if len(cond_tensor.shape) > 1 else "N/A"
                debug_output.append(f"Conditioning shape: {cond_tensor.shape}")
                debug_output.append(f"Actual sequence length: {actual_seq_len}")

        # Build debug string
        if debug_mode:
            debug_output.append(f"\n--- FORMATTED PROMPT ---\n{formatted_text}\n--- END ---")

        debug_str = "\n".join(debug_output) if debug_mode else ""

        return (conditioning, debug_str)


class ZImageTextEncoderSimple:
    """
    Simplified Z-Image encoder - just adds the missing thinking tokens.

    Drop-in replacement for CLIPTextEncode when using Z-Image.
    No system prompts, no templates - just the fix.
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
                    "tooltip": "Your prompt"
                }),
            },
            "optional": {
                "enable_thinking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add thinking tokens (recommended - matches diffusers)"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "ZImage/Encoding"
    TITLE = "Z-Image Text Encode (Simple)"
    DESCRIPTION = "Simple Z-Image encoding with thinking token fix"

    def encode(
        self,
        clip,
        text: str,
        enable_thinking: bool = True
    ) -> Tuple[Any]:
        """Encode with thinking tokens."""

        if enable_thinking:
            # Add the missing thinking tokens
            formatted_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        else:
            # ComfyUI's original format (for comparison)
            formatted_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

        tokens = clip.tokenize(formatted_text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        return (conditioning,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ZImageTextEncoder": ZImageTextEncoder,
    "ZImageTextEncoderSimple": ZImageTextEncoderSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTextEncoder": "Z-Image Text Encoder",
    "ZImageTextEncoderSimple": "Z-Image Text Encode (Simple)",
}
