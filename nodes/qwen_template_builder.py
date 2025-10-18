"""
Qwen Template Builder Node V2
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
                    "default": "",
                    "tooltip": "Your main prompt text"
                }),
                "template_mode": ([
                    "default_t2i",
                    "default_edit",
                    "inpainting",
                    "multi_image_edit",
                    "face_replacement",
                    "face_replacement_detailed",
                    "face_replacement_technical",
                    "identity_transfer",
                    "qwen_face_swap",
                    "qwen_identity_merge",
                    "structured_json_edit",
                    "xml_spatial_edit",
                    "natural_spatial_edit",
                    "artistic",
                    "photorealistic",
                    "minimal_edit",
                    "style_transfer",
                    "technical",
                    "custom_t2i",
                    "custom_edit",
                    "raw",
                    "show_all_prompts"
                ], {
                    "default": "default_edit",
                    "tooltip": "Choose template mode. Select 'show_all_prompts' to see all system prompts. Use custom_system field to override ANY template's system prompt."
                }),
                "custom_system": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Override system prompt - works with ANY template mode (leave empty to use default). This overrides the built-in system prompt for any selected template."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "system_prompt", "mode", "mode_info")
    FUNCTION = "build"
    CATEGORY = "QwenImage/Templates"
    TITLE = "Qwen Template Builder"
    DESCRIPTION = "Simple template builder with all options visible. Outputs mode for automatic encoder sync."

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
        "inpainting": {
            "system": "Replace or modify only the masked region according to the user's instruction. Preserve all other parts of the image exactly as they are. Blend the changes naturally with the surrounding context.",
            "vision": True,
            "mode": "inpainting"
        },
        "multi_image_edit": {
            "system": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
            "vision": True,
            "mode": "multi_image_edit",
            "use_picture_format": True
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
        },
        "structured_json_edit": {
            "system": "You are a precise image editor. Execute the JSON commands exactly as specified. Parse the JSON structure to understand the action, target, coordinates, and instructions. Maintain photorealistic quality and preserve elements marked as 'preserve'. Follow bounding box coordinates as absolute pixel locations.",
            "vision": True,
            "mode": "image_edit"
        },
        "xml_spatial_edit": {
            "system": "Process the spatial editing instructions in XML format. Each <region> element with data-bbox attributes defines a precise area to modify. Follow the <instruction> while preserving everything else. The data-bbox coordinates are absolute pixel locations in the format 'x1,y1,x2,y2'.",
            "vision": True,
            "mode": "image_edit"
        },
        "natural_spatial_edit": {
            "system": "Follow the coordinate-based editing instructions. When you see bounding box coordinates in brackets like [x1,y1,x2,y2], these specify exact pixel locations for modifications. Apply changes only within these regions while preserving all other areas.",
            "vision": True,
            "mode": "image_edit"
        },
        "face_replacement": {
            "system": "You are editing images based on user instructions. Analyze the provided images and follow this structured approach: The person from Picture 1 should replace the face and head of the person in Picture 2, while preserving all other elements of Picture 2 including body pose, clothing, hands, background, environment, and lighting. Generate a complete image that shows the full scene from Picture 2 with Picture 1's facial identity seamlessly integrated.",
            "vision": True,
            "mode": "image_edit",
            "use_picture_format": True
        },
        "face_replacement_detailed": {
            "system": "You are performing precise identity transfer between images. Follow this structured edit: Change the person from Picture 2 to have the exact facial features, skin tone, hair style, and head structure from Picture 1. Maintain Picture 2's complete scene composition including body posture, clothing, hand positions, background elements, and lighting conditions. Generate a full image that preserves Picture 2's context while replacing only the head and facial identity with Picture 1's features.",
            "vision": True,
            "mode": "image_edit",
            "use_picture_format": True
        },
        "face_replacement_technical": {
            "system": "FACE_EXTRACTION: isolate Picture_1.head[bbox(0.2,0.1,0.8,0.5)]; FEATURES: skin_tone=Picture_1.sample_rgb(), eye_shape=Picture_1.landmarks[36:48], nose=Picture_1.landmarks[27:36], mouth=Picture_1.landmarks[48:68], hair=Picture_1.region[above_forehead]; TARGET: Picture_2.preserve(all_except_head); OPERATION: Picture_2.head = Picture_1.head WITH blend_edges(neck, feather=20px) AND match_lighting(Picture_2.ambient);",
            "vision": True,
            "mode": "image_edit",
            "use_picture_format": True
        },
        "identity_transfer": {
            "system": "Execute complete identity transfer preserving Picture 1's person's exact appearance. Analysis Phase: Picture 1 - Capture complete facial identity including bone structure, features, skin tone, age markers, hair details. Picture 2 - Identify pose, clothing, environment, lighting angle and intensity. Synthesis Phase: Place Picture 1's person into Picture 2's exact scenario. The final image must look as if Picture 1's person was originally photographed in Picture 2's setting. Maintain photorealistic quality with proper shadow and light integration.",
            "vision": True,
            "mode": "image_edit",
            "use_picture_format": True
        },
        "qwen_face_swap": {
            "system": "Generate an image based on the following instructions: Change the person from Picture 2 to have the face from Picture 1. Keep all other elements from Picture 2 unchanged including body, clothing, pose, and background.",
            "vision": True,
            "mode": "image_edit",
            "use_picture_format": True
        },
        "qwen_identity_merge": {
            "system": "Generate an image that combines elements from multiple pictures: The face and head from Picture 1 should be placed on the body from Picture 2, maintaining Picture 2's pose, clothing, and environment. Ensure natural integration with consistent lighting and shadows.",
            "vision": True,
            "mode": "image_edit",
            "use_picture_format": True
        }
    }


    def build(self, prompt: str, template_mode: str, custom_system: str) -> Tuple[str, str, str, str]:
        """Output the raw prompt, system prompt, mode, and info for the encoder to use"""

        # Handle show all prompts mode - displays all available system prompts
        if template_mode == "show_all_prompts":
            prompt_list = "=== AVAILABLE SYSTEM PROMPTS ===\n\n"
            for name, template in self.TEMPLATES.items():
                prompt_list += f"[{name}]:\n{template['system']}\n" + "="*50 + "\n\n"
            prompt_list += "\nTO USE: Select a template above and optionally override with custom_system field"
            # Return display in system prompt field for viewing
            return (prompt, prompt_list, "text_to_image", "info|display_only")

        # Handle raw mode - no system prompt
        if template_mode == "raw":
            return (prompt, "", "text_to_image", "raw|no_template")

        # Handle custom modes
        if template_mode == "custom_t2i":
            system = custom_system if custom_system else "Generate the image based on the description."
            return (prompt, system, "text_to_image", "custom|text_to_image")

        if template_mode == "custom_edit":
            system = custom_system if custom_system else "Edit the image based on the instructions."
            return (prompt, system, "image_edit", "custom|image_edit")

        # Handle preset templates
        if template_mode in self.TEMPLATES:
            template = self.TEMPLATES[template_mode]
            # Use custom_system override if provided, otherwise use template default
            system_prompt = custom_system if custom_system else template["system"]
            mode = template['mode']
            mode_info = f"{template_mode}|{mode}"

            # Note: Picture format is now automatic in encoder for 2+ images

            return (prompt, system_prompt, mode, mode_info)

        # Fallback - no system prompt
        return (prompt, "", "text_to_image", "unknown|text_to_image")



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
