"""
HunyuanVideo 1.5 Text Encoder

Single encoder with template system for T2V. Outputs dual conditioning (positive/negative).

How it works:
- System + user text encoded together with attention
- System tokens influence user embeddings during encoding
- ComfyUI crops system tokens after encoding (handles crop_start correctly)
- User embeddings retain the "influence" of the system prompt

For basic encoding without templates, use ComfyUI's CLIPTextEncode node directly.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

# Try to import yaml for template parsing
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available, template metadata parsing limited")


class HunyuanVideoTextEncoder:
    """
    HunyuanVideo 1.5 Text Encoder with template system.
    Outputs both positive and negative conditioning for CFG.

    Features:
    - 23 video templates in nodes/templates/hunyuan_video_*.md
    - Custom system prompt support
    - Dual output (positive, negative) for direct KSampler connection
    - byT5 auto-triggered by quoted text (ComfyUI handles this)
    """

    DEFAULT_NEGATIVE = "low quality, blurry, distorted, artifacts, watermark, text, logo"

    def __init__(self):
        """Initialize with template loading."""
        self.templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.available_templates = self._load_available_templates()
        logger.info(f"[HunyuanVideoTextEncoder] Loaded {len(self.available_templates)} templates")

    def _load_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load available templates from templates/ directory."""
        templates = {}

        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return templates

        for filename in os.listdir(self.templates_dir):
            # Only load hunyuan_video templates
            if filename.startswith('hunyuan_video_') and filename.endswith('.md'):
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
        if not template_names:
            template_names = ["hunyuan_video_t2v"]

        return {
            "required": {
                "clip": ("CLIP",),
            },
            "optional": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Positive prompt - describe what you want"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "low quality, blurry, distorted, artifacts, watermark, text, logo",
                    "tooltip": "Negative prompt - describe what to avoid"
                }),
                "template_input": ("HUNYUAN_TEMPLATE", {
                    "tooltip": "Optional: Connect from HunyuanVideo Template Builder (overrides dropdown)"
                }),
                "template_preset": (["none"] + template_names, {
                    "default": "none",
                    "tooltip": "Video template from nodes/templates/ (ignored if template_input connected)"
                }),
                "custom_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Override system prompt (leave empty for default or preset)"
                }),
                "additional_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional instructions appended to template (e.g., 'always noir style', 'focus on hands')"
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show encoding details"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "debug_output")
    FUNCTION = "encode"
    CATEGORY = "HunyuanVideo/Encoding"
    TITLE = "HunyuanVideo Text Encoder"
    DESCRIPTION = "T2V encoder with templates. Outputs positive/negative for KSampler."

    def _adapt_template_for_video(self, system_prompt: str) -> str:
        """Adapt image-focused templates for video (word replacements)."""
        replacements = {
            "image": "video",
            "Image": "Video",
            "picture": "video",
            "Picture": "Video",
        }

        adapted = system_prompt
        for old, new in replacements.items():
            adapted = adapted.replace(old, new)

        # Add motion guidance if not present
        if "motion" not in adapted.lower() and "movement" not in adapted.lower():
            adapted += "\n\nDescribe the motion, camera movement, and temporal progression."

        return adapted

    def _encode_single(
        self,
        clip,
        text: str,
        system_prompt: str,
        uses_custom_template: bool,
        debug_output: List[str],
        debug_mode: bool,
        is_negative: bool = False
    ) -> Any:
        """Encode a single prompt (positive or negative) with template."""
        prompt_type = "negative" if is_negative else "positive"

        # Build formatted text with chat markers
        if uses_custom_template and system_prompt:
            formatted_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_text = text

        if debug_mode:
            debug_output.append(f"{prompt_type.capitalize()}: {len(formatted_text)} chars")

        # Tokenize and encode
        tokens = clip.tokenize(formatted_text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if debug_mode and conditioning and len(conditioning) > 0:
            cond_tensor = conditioning[0][0]
            if cond_tensor is not None:
                debug_output.append(f"{prompt_type.capitalize()} shape: {cond_tensor.shape}")

        return conditioning

    def encode(
        self,
        clip,
        text: str = "",
        negative_prompt: str = "",
        template_input: Optional[Dict[str, Any]] = None,
        template_preset: str = "none",
        custom_system_prompt: str = "",
        additional_instructions: str = "",
        debug_mode: bool = False
    ) -> Tuple[Any, Any, str]:
        """
        Encode both positive and negative prompts for HunyuanVideo.
        Both use the same system template for consistent embedding space.

        Priority order:
        1. template_input (from Template Builder) - if connected, uses its prompt and system
        2. custom_system_prompt - manual override
        3. template_preset dropdown - built-in template selection

        additional_instructions is always appended to whatever system prompt is used.
        """
        debug_output = []

        # Check if template_input is connected (from Template Builder)
        if template_input is not None:
            # Use prompt from template builder if text is empty
            if not text and template_input.get("prompt"):
                text = template_input["prompt"]
            # Use system prompt from template builder
            system_prompt = template_input.get("system_prompt", "")
            uses_custom_template = bool(system_prompt)
            template_name = template_input.get("template_name", "connected")
            debug_output.append(f"Template Builder: {template_name}")
        else:
            # Use default negative if empty
            if not negative_prompt:
                negative_prompt = self.DEFAULT_NEGATIVE

            # Determine system prompt from dropdown or custom
            system_prompt = ""
            uses_custom_template = False

            if custom_system_prompt:
                system_prompt = custom_system_prompt
                uses_custom_template = True
                debug_output.append(f"Custom system prompt ({len(custom_system_prompt)} chars)")
            elif template_preset != "none" and template_preset in self.available_templates:
                template_data = self.available_templates[template_preset]
                system_prompt = self._adapt_template_for_video(template_data['system_prompt'])
                uses_custom_template = True
                debug_output.append(f"Template: {template_preset}")
            else:
                debug_output.append("Using ComfyUI default template")

        # Append additional instructions if provided
        if additional_instructions.strip():
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n---\n\nAdditional Instructions:\n{additional_instructions.strip()}"
            else:
                system_prompt = additional_instructions.strip()
                uses_custom_template = True
            debug_output.append(f"Additional instructions: {len(additional_instructions)} chars")

        # Use default negative if still empty
        if not negative_prompt:
            negative_prompt = self.DEFAULT_NEGATIVE

        # Check for quoted text (byT5 trigger)
        if debug_mode and ('"' in text or '"' in text or '"' in text):
            debug_output.append("Quoted text found - byT5 will encode")

        # Encode positive
        positive_cond = self._encode_single(
            clip, text, system_prompt, uses_custom_template,
            debug_output, debug_mode, is_negative=False
        )

        # Encode negative (same template for consistent embedding space)
        negative_cond = self._encode_single(
            clip, negative_prompt, system_prompt, uses_custom_template,
            debug_output, debug_mode, is_negative=True
        )

        # Build debug string
        if debug_mode:
            debug_output.append(f"\n--- POSITIVE PROMPT ---\n{text}\n--- END ---")
            debug_output.append(f"\n--- NEGATIVE PROMPT ---\n{negative_prompt}\n--- END ---")
            if uses_custom_template and system_prompt:
                debug_output.append(f"\n--- SYSTEM PROMPT ---\n{system_prompt}\n--- END ---")

        debug_str = "\n".join(debug_output) if debug_mode else ""

        return (positive_cond, negative_cond, debug_str)


# Node registration
NODE_CLASS_MAPPINGS = {
    "HunyuanVideoTextEncoder": HunyuanVideoTextEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVideoTextEncoder": "HunyuanVideo Text Encoder",
}
