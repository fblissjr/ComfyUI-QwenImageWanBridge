"""
Qwen Template Builder Node V2
Simplified version with better Python-based logic
"""

import json
from typing import Dict, Any, Tuple

class QwenTemplateBuilderV2:
    """
    Simplified template builder that handles everything in Python
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape",
                    "tooltip": "Your main prompt text"
                }),
                "template_mode": ([
                    "default_t2i",
                    "default_edit", 
                    "artistic",
                    "photorealistic",
                    "minimal_edit",
                    "style_transfer",
                    "technical",
                    "custom_t2i",
                    "custom_edit",
                    "raw"
                ], {
                    "default": "default_edit",
                    "tooltip": "Choose template mode"
                }),
                "custom_system": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "tooltip": "Custom system prompt for custom modes"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("formatted_prompt", "use_with_encoder", "mode_info")
    FUNCTION = "build"
    CATEGORY = "QwenImage/Templates"
    TITLE = "Qwen Template Builder V2"
    DESCRIPTION = "Simple template builder with all options visible"
    
    # Template definitions
    TEMPLATES = {
        "default_t2i": {
            "system": "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:",
            "vision": False,
            "mode": "text_to_image"
        },
        "default_edit": {
            "system": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
            "vision": True,
            "mode": "image_edit"
        },
        "artistic": {
            "system": "You are an experimental artist. Break conventions. Be bold and creative. Interpret the prompt with artistic freedom.",
            "vision": False,
            "mode": "text_to_image"
        },
        "photorealistic": {
            "system": "You are a camera. Capture reality with perfect accuracy. No artistic interpretation. Focus on photorealistic details, proper lighting, and accurate proportions.",
            "vision": False,
            "mode": "text_to_image"
        },
        "minimal_edit": {
            "system": "Make only the specific changes requested. Preserve all other aspects of the original image exactly.",
            "vision": True,
            "mode": "image_edit"
        },
        "style_transfer": {
            "system": "Transform the image into the specified artistic style while preserving the original composition and subjects.",
            "vision": True,
            "mode": "image_edit"
        },
        "technical": {
            "system": "Generate technical diagrams and schematics. Use clean lines, proper labels, annotations, and professional technical drawing standards.",
            "vision": False,
            "mode": "text_to_image"
        }
    }
    
    def build(self, prompt: str, template_mode: str, custom_system: str) -> Tuple[str, bool, str]:
        """Build the formatted prompt"""
        
        # Handle raw mode
        if template_mode == "raw":
            return (prompt, False, "raw|no_template")
        
        # Handle custom modes
        if template_mode == "custom_t2i":
            formatted = self._format_template(custom_system, prompt, False)
            return (formatted, True, "custom|text_to_image")
        
        if template_mode == "custom_edit":
            formatted = self._format_template(custom_system, prompt, True)
            return (formatted, True, "custom|image_edit")
        
        # Handle preset templates
        if template_mode in self.TEMPLATES:
            template = self.TEMPLATES[template_mode]
            formatted = self._format_template(
                template["system"],
                prompt,
                template["vision"]
            )
            return (formatted, True, f"{template_mode}|{template['mode']}")
        
        # Fallback
        return (prompt, False, "unknown|text_to_image")
    
    def _format_template(self, system: str, prompt: str, include_vision: bool) -> str:
        """Format the chat template"""
        
        result = f"<|im_start|>system\n{system}<|im_end|>\n"
        result += f"<|im_start|>user\n"
        
        if include_vision:
            result += "<|vision_start|><|image_pad|><|vision_end|>"
        
        result += f"{prompt}<|im_end|>\n"
        result += f"<|im_start|>assistant\n"
        
        return result


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