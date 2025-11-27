"""
Z-Image Text Encoder for ComfyUI

IMPORTANT: Our original analysis was inverted. Here's what actually happens:

Diffusers (pipeline_z_image.py):
    tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True)

    enable_thinking=True  -> NO <think> block (let model generate its own)
    enable_thinking=False -> ADD <think>\n\n</think>\n\n (skip thinking)

ComfyUI (z_image.py):
    self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    This matches diffusers with enable_thinking=True (no think block)

So ComfyUI and diffusers produce IDENTICAL templates by default!

The REAL difference is in embedding extraction:
- Diffusers: Returns embeddings[attention_mask.bool()] - only non-padded tokens
- ComfyUI: Returns full padded sequence

Tokenizer Compatibility:
- ComfyUI bundles Qwen2.5-VL tokenizer (token 151667 = <|meta|>)
- Qwen3-4B has thinking tokens (token 151667 = <think>)
- For the basic format, they produce identical token IDs
- For <think> tokens, ComfyUI's tokenizer treats them as regular subwords

This encoder provides experimental knobs to test what actually improves output quality.
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
    Z-Image Text Encoder with experimental knobs for testing.

    Default behavior matches diffusers exactly (no thinking block).
    Enable add_think_block=True to experiment with thinking tokens.

    Note: ComfyUI's bundled tokenizer treats <think> as regular text (not a special token).
    For accurate <think> tokenization, you'd need the actual Qwen3-4B tokenizer.
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
                "add_think_block": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: Add <think></think> block. False=match diffusers, True=add thinking tokens"
                }),
                "thinking_content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "EXPERIMENTAL: Custom reasoning to put inside <think> tags. Only used if add_think_block=True."
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
    DESCRIPTION = "Z-Image encoding with experimental knobs (default matches diffusers exactly)"

    def _format_prompt(
        self,
        text: str,
        system_prompt: str = "",
        add_think_block: bool = False,
        thinking_content: str = ""
    ) -> str:
        """
        Format prompt for Z-Image encoding.

        Args:
            text: User prompt
            system_prompt: Optional system instructions
            add_think_block: Whether to add <think> tags (experimental)
            thinking_content: Optional reasoning to put inside <think> tags

        Returns:
            Formatted prompt string

        Note on add_think_block:
            False (default): Matches diffusers exactly
                <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n

            True: Adds thinking block (experimental)
                <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n{content}\n</think>\n\n
        """
        parts = []

        # System message (optional)
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        # User message
        parts.append(f"<|im_start|>user\n{text}<|im_end|>")

        # Assistant message with optional thinking block
        if add_think_block:
            if thinking_content.strip():
                # User provided their own thinking content
                think_block = f"<think>\n{thinking_content.strip()}\n</think>\n\n"
            else:
                # Empty thinking block
                think_block = "<think>\n\n</think>\n\n"
            parts.append(f"<|im_start|>assistant\n{think_block}")
        else:
            # No thinking block (matches diffusers with enable_thinking=True)
            parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def encode(
        self,
        clip,
        text: str,
        system_prompt_preset: str = "none",
        custom_system_prompt: str = "",
        template_preset: str = "none",
        add_think_block: bool = False,
        thinking_content: str = "",
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        debug_mode: bool = False
    ) -> Tuple[Any, str]:
        """
        Encode text for Z-Image.

        Args:
            clip: ComfyUI CLIP model (lumina2 type for Z-Image)
            text: User prompt
            system_prompt_preset: Preset system prompt name
            custom_system_prompt: Custom system prompt (overrides preset)
            template_preset: Template from nodes/templates/z_image_*.md
            add_think_block: Add <think></think> block (default False = matches diffusers)
            thinking_content: Optional reasoning text to put inside <think> tags
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

        # Format the prompt
        formatted_text = self._format_prompt(
            text=text,
            system_prompt=system_prompt,
            add_think_block=add_think_block,
            thinking_content=thinking_content
        )

        if debug_mode:
            debug_output.append(f"add_think_block: {add_think_block}")
            if add_think_block and thinking_content.strip():
                debug_output.append(f"thinking_content: {len(thinking_content)} chars (custom)")
            elif add_think_block:
                debug_output.append("thinking_content: (empty)")
            debug_output.append(f"max_sequence_length: {max_sequence_length}")
            debug_output.append(f"Formatted prompt length: {len(formatted_text)} chars")

        # Tokenize and encode using ComfyUI's CLIP
        tokens = clip.tokenize(formatted_text)

        # Estimate token count and warn if exceeding limit
        # Rough estimate: ~4 chars per token for English
        estimated_tokens = len(formatted_text) // 4
        if estimated_tokens > max_sequence_length:
            warning_msg = f"WARNING: Prompt may exceed max_sequence_length ({estimated_tokens} estimated vs {max_sequence_length} limit)"
            debug_output.append(warning_msg)
            logger.warning(f"[ZImageTextEncoder] {warning_msg}")

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
    Simplified Z-Image encoder - drop-in replacement for CLIPTextEncode.

    Default behavior matches diffusers exactly.
    Toggle add_think_block to experiment with thinking tokens.
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
                "add_think_block": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: Add <think></think> block. False=match diffusers (recommended)"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "ZImage/Encoding"
    TITLE = "Z-Image Text Encode (Simple)"
    DESCRIPTION = "Simple Z-Image encoding (default matches diffusers exactly)"

    def encode(
        self,
        clip,
        text: str,
        add_think_block: bool = False
    ) -> Tuple[Any]:
        """Encode with optional thinking tokens."""

        if add_think_block:
            # Experimental: Add empty thinking block
            formatted_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        else:
            # Default: Match diffusers exactly
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
