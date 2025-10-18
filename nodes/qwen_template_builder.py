"""
Qwen Template Builder Node V2
Loads templates from nodes/templates/*.md files
"""

import json
from typing import Dict, Any, Tuple
from .template_loader import get_template_loader

class QwenTemplateBuilderV2:
    """
    Simplified template builder that handles everything in Python
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Load template names from files
        loader = get_template_loader()
        template_names = loader.get_template_names()

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your main prompt text"
                }),
                "template_mode": (template_names, {
                    "default": "default_edit",
                    "tooltip": "Choose template mode. Templates loaded from nodes/templates/*.md files. Use custom_system field to edit any template."
                }),
                "custom_system": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Override system prompt - works with ANY template mode (leave empty to use default). JavaScript UI auto-fills this when you select a template."
                }),
            }
        }

    RETURN_TYPES = ("QWEN_TEMPLATE", "STRING")
    RETURN_NAMES = ("template_output", "debug_info")
    FUNCTION = "build"
    CATEGORY = "QwenImage/Templates"
    TITLE = "Qwen Template Builder"
    DESCRIPTION = "Template builder using nodes/templates/*.md files. Single output contains everything."


    def build(self, prompt: str, template_mode: str, custom_system: str) -> Tuple[Dict[str, Any], str]:
        """Output template dict with all settings, plus debug info"""

        # Load templates from files
        loader = get_template_loader()
        template = loader.get_template(template_mode)

        if not template:
            # Fallback if template not found
            output = {
                "prompt": prompt,
                "system_prompt": "",
                "mode": "text_to_image",
                "template_name": template_mode
            }
            debug = f"Template: {template_mode} (not found - fallback to text_to_image)"
            return (output, debug)

        # Handle raw mode - no system prompt
        if template.get("no_template", False):
            output = {
                "prompt": prompt,
                "system_prompt": "",
                "mode": template["mode"],
                "template_name": template_mode
            }
            debug = f"Template: {template_mode} (raw - no system prompt)\nMode: {template['mode']}\nPrompt: {prompt}"
            return (output, debug)

        # Use custom_system override if provided, otherwise use template default
        system_prompt = custom_system if custom_system else template["system"]
        mode = template["mode"]

        output = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "mode": mode,
            "template_name": template_mode,
            "experimental": template.get("experimental", False)
        }

        # Build debug info
        debug = f"=== TEMPLATE DEBUG ===\n"
        debug += f"Template: {template_mode}\n"
        debug += f"Mode: {mode}\n"
        if template.get("experimental"):
            debug += f"Status: EXPERIMENTAL\n"
        debug += f"\n=== SYSTEM PROMPT ===\n{system_prompt}\n"
        debug += f"\n=== USER PROMPT ===\n{prompt}\n"
        debug += f"\n=== SETTINGS ===\n"
        debug += f"Vision tokens: {'Yes' if template.get('vision', False) else 'No'}\n"
        debug += f"Picture format: {'Yes' if template.get('use_picture_format', False) else 'No'}\n"

        return (output, debug)



class QwenTemplateConnector:
    """
    Connects template builder to encoder with proper settings
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "formatted_prompt": ("STRING", {"forceInput": True}),
                "use_with_encoder": ("BOOLEAN", {"forceInput": True}),
                "mode_info": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("text_for_encoder", "encoder_mode", "apply_template", "token_removal")
    FUNCTION = "connect"
    CATEGORY = "QwenImage/Templates"
    TITLE = "Qwen Template Connector"
    DESCRIPTION = "Properly connects template to encoder"

    def connect(self, formatted_prompt: str, use_with_encoder: bool, mode_info: str) -> Tuple[str, str, bool, str]:
        """
        Set up proper encoder settings based on template output
        """

        # Parse mode info
        parts = mode_info.split("|")
        template_type = parts[0] if len(parts) > 0 else "unknown"
        mode = parts[1] if len(parts) > 1 else "text_to_image"

        # Determine encoder settings
        if not use_with_encoder:
            # Raw mode - pass through as-is
            return (formatted_prompt, mode, False, "none")
        else:
            # Template already applied - don't reapply
            apply_template = False
            token_removal = "none"  # Template already formatted correctly

            return (formatted_prompt, mode, apply_template, token_removal)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenTemplateBuilderV2": QwenTemplateBuilderV2,
    "QwenTemplateConnector": QwenTemplateConnector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTemplateBuilderV2": "Qwen Template Builder V2",
    "QwenTemplateConnector": "Qwen Template Connector",
}
