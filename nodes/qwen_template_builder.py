"""
Qwen Template Builder Node
Simplified version with better Python-based logic
"""

import json
import logging
import torch
from typing import Dict, Any, Tuple
import folder_paths

logger = logging.getLogger(__name__)

# Try to import transformers for direct model access
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available for direct Qwen2.5-VL generation")

class QwenTemplateBuilder:
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
                    "conversation_generation",
                    "custom_t2i",
                    "custom_edit",
                    "raw"
                ], {
                    "default": "default_edit",
                    "tooltip": "Choose template mode"
                }),
                "custom_system": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom system prompt for custom modes"
                }),
            },
            "optional": {
                "qwen_model": ("QWEN_MODEL", {
                    "tooltip": "Connect QwenUnifiedLoader for conversation generation mode"
                }),
                "clip": ("CLIP", {
                    "tooltip": "Connect QwenVLCLIPLoader (legacy) for conversation generation mode"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Reference images for conversation generation (connects to Multi-Reference Handler or direct images)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("formatted_prompt", "use_with_encoder", "mode_info")
    FUNCTION = "build"
    CATEGORY = "QwenImage/Templates"
    TITLE = "Qwen Template Builder"
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

    def build(self, prompt: str, template_mode: str, custom_system: str, 
              qwen_model=None, clip=None, images=None) -> Tuple[str, bool, str]:
        """Build the formatted prompt"""

        # Handle raw mode
        if template_mode == "raw":
            return (prompt, False, "raw|no_template")

        # Handle conversation generation mode
        if template_mode == "conversation_generation":
            # Check for either unified model or legacy CLIP
            model_to_use = None
            
            if qwen_model is not None:
                model_to_use = qwen_model
            elif clip is not None:
                # Convert CLIP to unified model interface (fallback to template)
                logger.info("[Template Builder] Using CLIP input, falling back to template generation")
                return (self._create_smart_template_conversation(prompt, images), True, "conversation_template|image_edit")
            else:
                return ("Error: Connect QwenUnifiedLoader or QwenVLCLIPLoader to use conversation generation mode", False, "error|conversation_generation")
            
            try:
                conversation = self._generate_conversation(model_to_use, prompt, images)
                return (conversation, True, "conversation_generated|image_edit")
            except Exception as e:
                return (f"Conversation generation failed: {str(e)}", False, "error|conversation_generation")

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

    def _generate_conversation(self, qwen_model, seed_prompt: str, images=None) -> str:
        """Generate a natural conversation using QwenUnifiedModel"""
        
        # Use the unified model for generation
        try:
            return self._generate_with_qwen_model(qwen_model, seed_prompt, images)
        except Exception as e:
            logger.warning(f"Unified model generation failed: {e}")
        
        # Fallback: Smart template that incorporates the seed prompt
        logger.info("Using smart template generation as fallback")
        return self._create_smart_template_conversation(seed_prompt, images)
    
    def _generate_with_qwen_model(self, qwen_model, seed_prompt: str, images=None) -> str:
        """Generate conversation using the already-loaded QwenUnifiedModel"""
        
        # Create meta-prompt for conversation generation
        meta_prompt = f"""<|im_start|>system
You are creating a natural conversation about image generation. Look at the provided images and generate a realistic dialogue.

Goal: {seed_prompt}

Create a conversation where:
- User mentions wanting to create something with reference images
- Assistant asks what they see and want to create  
- User explains their goal
- Assistant references specific visual details from the images
- Conversation builds understanding and ends ready for generation

Format with proper <|im_start|>user and <|im_start|>assistant tags.<|im_end|>
<|im_start|>user
I want to create something using these reference images<|im_end|>
<|im_start|>assistant
"""
        
        # Use the unified model's generate method
        try:
            generated_text = qwen_model.generate(
                meta_prompt,
                images=images,
                max_new_tokens=400,
                temperature=0.8,
                do_sample=True,
                top_p=0.9
            )
            
            # Combine and clean
            full_conversation = meta_prompt + generated_text
            return self._clean_generated_conversation(full_conversation)
            
        except Exception as e:
            logger.error(f"Unified model generation failed: {e}")
            raise
    
    def _get_qwen_model_from_clip(self, clip):
        """Try to extract the actual Qwen2.5-VL model from ComfyUI's CLIP wrapper"""
        
        # ComfyUI wraps models in various ways, try to access the underlying model
        if hasattr(clip, 'cond_stage_model'):
            model = clip.cond_stage_model
        elif hasattr(clip, 'model'):
            model = clip.model
        else:
            return None
        
        # Look for the actual transformers model
        if hasattr(model, 'model') and hasattr(model.model, 'generate'):
            return model.model  # This should be the transformers Qwen2VLForConditionalGeneration
        elif hasattr(model, 'generate'):
            return model
        
        return None
    
    def _generate_with_model(self, qwen_model, seed_prompt: str, images):
        """Use actual Qwen2.5-VL model to generate conversation"""
        
        # Meta-prompt for conversation generation
        meta_prompt = f"""<|im_start|>system
You are helping create a natural conversation between a user and assistant about image generation. Generate a realistic back-and-forth conversation.

The user's goal: "{seed_prompt}"

Create conversation that:
- References specific visual details you see
- Asks clarifying questions about the images  
- Builds understanding through dialogue
- Ends with <|im_start|>assistant (no <|im_end|>) ready for generation<|im_end|>
<|im_start|>user
I want to create something with these reference images<|im_end|>
<|im_start|>assistant
"""

        try:
            # Prepare inputs for generation
            inputs = qwen_model.tokenizer(
                meta_prompt, 
                images=images if images is not None else None,
                return_tensors="pt"
            )
            
            # Generate conversation
            with torch.no_grad():
                outputs = qwen_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=qwen_model.tokenizer.eos_token_id
                )
            
            # Decode the generated conversation
            generated_text = qwen_model.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=False
            )
            
            # Clean up and format
            full_conversation = meta_prompt + generated_text
            return self._clean_generated_conversation(full_conversation)
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise e
    
    def _clean_generated_conversation(self, generated_text: str) -> str:
        """Clean up generated conversation"""
        
        # Find the start of the actual conversation
        start_marker = "<|im_start|>user\nI want to create something"
        if start_marker in generated_text:
            conversation_start = generated_text.find(start_marker)
            cleaned = generated_text[conversation_start:]
        else:
            cleaned = generated_text
        
        # Ensure it ends with assistant ready for generation
        if not cleaned.endswith("<|im_start|>assistant"):
            if cleaned.endswith("<|im_end|>"):
                cleaned = cleaned[:-10] + "<|im_start|>assistant"
            else:
                cleaned += "\n<|im_start|>assistant"
        
        return cleaned
    
    def _create_smart_template_conversation(self, seed_prompt: str, images) -> str:
        """Create an enhanced template conversation that's more specific to the seed prompt"""
        
        # Analyze seed prompt for key elements
        has_style_reference = any(word in seed_prompt.lower() for word in ['style', 'art', 'look', 'aesthetic'])
        has_character = any(word in seed_prompt.lower() for word in ['portrait', 'person', 'character', 'face'])
        has_composition = any(word in seed_prompt.lower() for word in ['sitting', 'standing', 'pose', 'position'])
        
        # Create contextual conversation
        conversation = f"""<|im_start|>user
I want to create something with these reference images<|im_end|>
<|im_start|>assistant"""
        
        if images is not None:
            conversation += f"""
I can see your reference images. What specifically would you like me to help you create?<|im_end|>
<|im_start|>user
{seed_prompt}<|im_end|>
<|im_start|>assistant"""
            
            if has_style_reference:
                conversation += f"""
That's a great idea! I can see some interesting visual styles in your reference images. Should I take the artistic style and color palette from these images for your creation?<|im_end|>
<|im_start|>user
Yes, exactly! Use the visual style from the references<|im_end|>
<|im_start|>assistant"""
            
            if has_character:
                conversation += f"""
Perfect! And I notice you want to create a portrait. Should I incorporate any specific visual elements or characters I see in these reference images into the composition?<|im_end|>
<|im_start|>user
Yes, blend the visual elements from both images appropriately<|im_end|>
<|im_start|>assistant"""
        
        else:
            conversation += f"""
I'd be happy to help! What specifically would you like me to create?<|im_end|>
<|im_start|>user
{seed_prompt}<|im_end|>
<|im_start|>assistant"""
        
        # Final readiness
        conversation += """
Excellent! I understand exactly what you're looking for. I'll create that for you now, incorporating all the elements we discussed<|im_end|>
<|im_start|>assistant"""
        
        return conversation
    
    def _create_template_conversation(self, seed_prompt: str) -> str:
        """Create a template conversation as fallback until we implement full generation"""
        
        conversation = f"""<|im_start|>user
I want to create something with these reference images<|im_end|>
<|im_start|>assistant
I can see your reference images. What specifically would you like me to help you create?<|im_end|>
<|im_start|>user
{seed_prompt}<|im_end|>
<|im_start|>assistant
That sounds interesting! Let me look at your reference images more carefully. I can see some great visual elements we could work with. Should I incorporate the style and elements from these images into your creation?<|im_end|>
<|im_start|>user
Yes, exactly! Use the visual style and any relevant elements you see<|im_end|>
<|im_start|>assistant"""
        
        return conversation


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
    "QwenTemplateBuilder": QwenTemplateBuilder,
    "QwenTemplateConnector": QwenTemplateConnector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTemplateBuilder": "Qwen Template Builder",
    "QwenTemplateConnector": "Qwen Template Connector",
}
