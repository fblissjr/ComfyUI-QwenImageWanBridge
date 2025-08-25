"""
Qwen Template Builder Node
Interactive node for building custom Qwen prompts with template presets
"""

import json
from typing import Dict, Any, Tuple

class QwenTemplateBuilder:
    """
    Helper node for building custom Qwen prompts with templates
    Provides presets and visual feedback for template construction
    """
    
    # Template presets matching the JS file
    TEMPLATES = {
        "default_t2i": {
            "name": "Default Text-to-Image",
            "system": "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:",
            "has_vision": False
        },
        "default_edit": {
            "name": "Default Image Edit", 
            "system": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
            "has_vision": True
        },
        "artistic": {
            "name": "Artistic Freedom",
            "system": "You are an experimental artist. Break conventions. Be bold and creative. Interpret the prompt with artistic freedom.",
            "has_vision": False
        },
        "photorealistic": {
            "name": "Photorealistic",
            "system": "You are a camera. Capture reality with perfect accuracy. No artistic interpretation. Focus on photorealistic details, proper lighting, and accurate proportions.",
            "has_vision": False
        },
        "minimal_edit": {
            "name": "Minimal Edit",
            "system": "Make only the specific changes requested. Preserve all other aspects of the original image exactly.",
            "has_vision": True
        },
        "style_transfer": {
            "name": "Style Transfer",
            "system": "Transform the image into the specified artistic style while preserving the original composition and subjects.",
            "has_vision": True
        },
        "technical": {
            "name": "Technical/Diagram",
            "system": "Generate technical diagrams and schematics. Use clean lines, proper labels, annotations, and professional technical drawing standards.",
            "has_vision": False
        },
        "custom": {
            "name": "Custom Template",
            "system": "",  # User provides this
            "has_vision": False  # User chooses
        },
        "raw": {
            "name": "Raw Prompt (No Template)",
            "system": None,  # No template
            "has_vision": False
        }
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        template_names = list(cls.TEMPLATES.keys())
        template_display = [cls.TEMPLATES[k]["name"] for k in template_names]
        
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape",
                    "tooltip": "Your main prompt text"
                }),
                "template_preset": (template_names, {
                    "default": "default_edit",
                    "tooltip": "Choose a template preset or 'custom' to build your own"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful image generator.",
                    "tooltip": "Custom system prompt (only used with 'custom' template)",
                    "dynamicPrompts": False
                }),
                "include_vision_tokens": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include vision tokens for image editing (custom template only)"
                }),
                "show_special_tokens": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Display special tokens in the output for learning"
                }),
            },
            "hidden": {
                "template_preview": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Preview of the generated template"
                }),
                "token_reference": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Special tokens reference"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("formatted_prompt", "template_info", "has_vision_tokens")
    FUNCTION = "build_template"
    CATEGORY = "QwenImage/Templates"
    TITLE = "Qwen Template Builder"
    DESCRIPTION = "Build custom prompts with template presets and visual feedback"
    
    def build_template(self, prompt: str, template_preset: str,
                       system_prompt: str = "", include_vision_tokens: bool = False,
                       show_special_tokens: bool = True,
                       **kwargs) -> Tuple[str, str, bool]:
        """
        Build the formatted prompt based on selected template
        
        Returns:
            - formatted_prompt: The complete formatted prompt
            - template_info: JSON info about the template used
            - has_vision_tokens: Whether vision tokens are included
        """
        
        template = self.TEMPLATES.get(template_preset, self.TEMPLATES["default_t2i"])
        
        # Handle raw prompt (no template)
        if template_preset == "raw":
            template_info = json.dumps({
                "preset": "raw",
                "name": template["name"],
                "has_template": False,
                "has_vision": False
            }, indent=2)
            return (prompt, template_info, False)
        
        # Handle custom template
        if template_preset == "custom":
            if not system_prompt:
                system_prompt = "You are a helpful image generator."
            
            formatted = self._build_chat_template(
                system_prompt, 
                prompt,
                include_vision_tokens
            )
            
            template_info = json.dumps({
                "preset": "custom",
                "name": "Custom Template",
                "has_template": True,
                "has_vision": include_vision_tokens,
                "system": system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
            }, indent=2)
            
            return (formatted, template_info, include_vision_tokens)
        
        # Use preset template
        system = template["system"]
        has_vision = template["has_vision"]
        
        formatted = self._build_chat_template(system, prompt, has_vision)
        
        template_info = json.dumps({
            "preset": template_preset,
            "name": template["name"],
            "has_template": True,
            "has_vision": has_vision,
            "system": system[:100] + "..." if len(system) > 100 else system
        }, indent=2)
        
        return (formatted, template_info, has_vision)
    
    def _build_chat_template(self, system: str, user_prompt: str, include_vision: bool) -> str:
        """Build the chat-formatted template"""
        
        template = f"<|im_start|>system\n{system}<|im_end|>\n"
        template += f"<|im_start|>user\n"
        
        if include_vision:
            template += "<|vision_start|><|image_pad|><|vision_end|>"
        
        template += f"{user_prompt}<|im_end|>\n"
        template += f"<|im_start|>assistant\n"
        
        return template

class QwenTokenInfo:
    """
    Display node showing Qwen special tokens and their usage
    Educational helper node
    """
    
    TOKEN_INFO = {
        "<|im_start|>": "Marks the beginning of a message from a specific role",
        "<|im_end|>": "Marks the end of a message",
        "<|vision_start|>": "Indicates the start of vision input (for image editing)",
        "<|image_pad|>": "Placeholder where image data will be inserted",
        "<|vision_end|>": "Indicates the end of vision input",
        "system": "Role for system instructions that guide behavior",
        "user": "Role for user input/instructions",
        "assistant": "Role marker for model responses"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "display_mode": (["compact", "detailed", "examples"], {
                    "default": "detailed",
                    "tooltip": "How to display token information"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("token_reference",)
    FUNCTION = "get_info"
    CATEGORY = "QwenImage/Templates"
    TITLE = "Qwen Token Reference"
    DESCRIPTION = "Display reference for Qwen special tokens"
    
    def get_info(self, display_mode: str) -> Tuple[str]:
        """Generate token reference based on display mode"""
        
        if display_mode == "compact":
            info = "QWEN SPECIAL TOKENS:\n"
            for token, desc in self.TOKEN_INFO.items():
                if token.startswith("<"):
                    info += f"{token}\n"
                    
        elif display_mode == "detailed":
            info = "QWEN SPECIAL TOKENS REFERENCE:\n" + "="*40 + "\n\n"
            for token, desc in self.TOKEN_INFO.items():
                info += f"{token:<20} {desc}\n"
            info += "\n" + "="*40 + "\n"
            info += "\nUSAGE: These tokens structure the conversation format.\n"
            info += "Vision tokens are only used in image editing mode.\n"
            
        else:  # examples
            info = "QWEN TOKEN EXAMPLES:\n" + "="*40 + "\n\n"
            info += "TEXT-TO-IMAGE:\n"
            info += "<|im_start|>system\n[System instructions]<|im_end|>\n"
            info += "<|im_start|>user\n[Your prompt]<|im_end|>\n"
            info += "<|im_start|>assistant\n\n"
            info += "-"*40 + "\n\n"
            info += "IMAGE EDITING:\n"
            info += "<|im_start|>system\n[System instructions]<|im_end|>\n"
            info += "<|im_start|>user\n"
            info += "<|vision_start|><|image_pad|><|vision_end|>[Edit instructions]<|im_end|>\n"
            info += "<|im_start|>assistant\n"
        
        return (info,)

class QwenPromptFormatter:
    """
    Simple formatter that connects to QwenVLTextEncoder
    Provides a bridge between template builder and encoder
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "formatted_prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": "Pre-formatted prompt from template builder"
                }),
                "has_vision_tokens": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether the prompt contains vision tokens"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("text", "is_edit_mode")
    FUNCTION = "format"
    CATEGORY = "QwenImage/Templates"
    TITLE = "Qwen Prompt Formatter"
    DESCRIPTION = "Bridge between template builder and text encoder"
    
    def format(self, formatted_prompt: str, has_vision_tokens: bool) -> Tuple[str, bool]:
        """
        Pass through the formatted prompt and determine mode
        
        Returns:
            - text: The formatted prompt to pass to encoder
            - is_edit_mode: Whether to use image_edit mode
        """
        # Determine if we should use edit mode based on vision tokens
        is_edit_mode = has_vision_tokens
        
        return (formatted_prompt, is_edit_mode)

# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenTemplateBuilder": QwenTemplateBuilder,
    "QwenTokenInfo": QwenTokenInfo,
    "QwenPromptFormatter": QwenPromptFormatter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTemplateBuilder": "Qwen Template Builder",
    "QwenTokenInfo": "Qwen Token Reference",
    "QwenPromptFormatter": "Qwen Prompt Formatter",
}