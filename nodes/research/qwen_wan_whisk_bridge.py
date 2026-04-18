"""
Whisk-style Bridge: Image → VLM → Editable Text → Video
The most reliable pipeline, following Google Whisk's approach
"""

import torch
from typing import Dict, Optional, Tuple

class QwenWANWhiskBridge:
    """
    Google Whisk-style pipeline:
    1. Qwen analyzes image → generates text description
    2. User can edit the text
    3. Text → embeddings → WAN generation
    
    This avoids direct latent transfer issues entirely!
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "vlm_prompt": ("STRING", {
                    "default": "Describe this image in detail for video generation. Include subject, action, setting, mood, and visual style.",
                    "multiline": True
                }),
                "auto_enhance": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vlm_context": ("VLM_CONTEXT",),  # From ShrugPrompter
                "manual_description": ("STRING", {"multiline": True}),  # Override VLM
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "GUIDE")
    RETURN_NAMES = ("description", "enhanced_prompt", "generation_guide")
    FUNCTION = "analyze"
    CATEGORY = "QwenWANBridge/Whisk"
    
    def analyze(self, image, vlm_prompt, auto_enhance, 
                vlm_context=None, manual_description=None):
        
        # If manual description provided, use it directly
        if manual_description and manual_description.strip():
            description = manual_description.strip()
            source = "manual"
        else:
            # This would normally call VLM, but for now return template
            description = "[VLM would analyze image here with prompt: {}]".format(vlm_prompt[:50])
            source = "vlm_placeholder"
        
        # Auto-enhance for video generation
        if auto_enhance:
            enhanced = self.enhance_for_video(description)
        else:
            enhanced = description
        
        # Create generation guide
        guide = {
            "description": description,
            "enhanced": enhanced,
            "source": source,
            "suggestions": self.get_suggestions(description),
            "wan_optimized": self.optimize_for_wan(enhanced),
        }
        
        return (description, enhanced, guide)
    
    def enhance_for_video(self, description):
        """Enhance description for better video generation"""
        
        # Add video-specific keywords
        video_hints = []
        
        # Motion hints
        if "standing" in description.lower():
            video_hints.append("slowly swaying")
        elif "sitting" in description.lower():
            video_hints.append("gentle breathing motion")
            
        # Camera hints
        if "portrait" in description.lower() or "face" in description.lower():
            video_hints.append("subtle zoom in")
        elif "landscape" in description.lower():
            video_hints.append("slow pan across scene")
        
        # Quality hints
        video_hints.extend(["smooth motion", "high quality", "cinematic"])
        
        # Combine
        if video_hints:
            enhanced = f"{description}, {', '.join(video_hints)}"
        else:
            enhanced = description
            
        return enhanced
    
    def optimize_for_wan(self, prompt):
        """Optimize prompt specifically for WAN 2.2"""
        
        # WAN prefers certain phrase structures
        optimizations = {
            # Style optimizations
            "realistic": "photorealistic, detailed",
            "cartoon": "animated style, cel shaded",
            "anime": "anime style, 2D animation",
            
            # Motion optimizations  
            "moving": "smooth motion, natural movement",
            "walking": "walking forward, steady gait",
            "talking": "speaking, lip sync motion",
            
            # Quality markers
            "high quality": "masterpiece, best quality, high resolution",
            "detailed": "intricate details, sharp focus",
        }
        
        optimized = prompt
        for key, value in optimizations.items():
            if key in optimized.lower():
                optimized = optimized.lower().replace(key, value)
        
        return optimized
    
    def get_suggestions(self, description):
        """Suggest edits to improve generation"""
        
        suggestions = []
        
        # Check for motion descriptors
        motion_words = ["moving", "walking", "running", "dancing", "turning"]
        if not any(word in description.lower() for word in motion_words):
            suggestions.append("Add motion: 'slowly turning', 'gently swaying', etc.")
        
        # Check for camera movement
        camera_words = ["zoom", "pan", "track", "dolly", "rotate"]
        if not any(word in description.lower() for word in camera_words):
            suggestions.append("Add camera: 'slow zoom in', 'pan left to right', etc.")
        
        # Check for temporal markers
        temporal_words = ["then", "gradually", "slowly", "quickly", "suddenly"]
        if not any(word in description.lower() for word in temporal_words):
            suggestions.append("Add timing: 'gradually', 'slowly', 'then', etc.")
        
        return suggestions


class WhiskTextEditor:
    """
    Interactive text editor for Whisk-style pipeline
    Allows editing VLM output before generation
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_text": ("STRING", {"multiline": True}),
                "edit_mode": (["append", "prepend", "replace", "enhance"], {"default": "enhance"}),
                "edit_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "style_preset": ([
                    "none",
                    "cinematic", 
                    "documentary",
                    "anime",
                    "noir",
                    "vibrant",
                    "monochrome"
                ], {"default": "none"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("edited_text", "comparison")
    FUNCTION = "edit"
    CATEGORY = "QwenWANBridge/Whisk"
    
    def edit(self, original_text, edit_mode, edit_text, style_preset="none"):
        
        if edit_mode == "append":
            edited = f"{original_text} {edit_text}"
            
        elif edit_mode == "prepend":
            edited = f"{edit_text} {original_text}"
            
        elif edit_mode == "replace":
            edited = edit_text if edit_text else original_text
            
        elif edit_mode == "enhance":
            # Smart enhancement based on content
            enhancements = []
            
            # Add style preset
            if style_preset != "none":
                style_map = {
                    "cinematic": "cinematic lighting, dramatic composition, film grain",
                    "documentary": "realistic, handheld camera, natural lighting",
                    "anime": "anime style, cel shaded, vibrant colors",
                    "noir": "film noir style, high contrast, dramatic shadows",
                    "vibrant": "vibrant colors, high saturation, dynamic",
                    "monochrome": "black and white, monochromatic, high contrast",
                }
                enhancements.append(style_map.get(style_preset, ""))
            
            # Add edit text as enhancement
            if edit_text:
                enhancements.append(edit_text)
            
            # Combine
            if enhancements:
                edited = f"{original_text}, {', '.join(enhancements)}"
            else:
                edited = original_text
        
        # Create comparison
        comparison = f"""ORIGINAL:
{original_text}

EDITED ({edit_mode}):
{edited}

CHANGES:
- Mode: {edit_mode}
- Style: {style_preset}
- Added: {edit_text if edit_text else 'nothing'}"""
        
        return (edited, comparison)


class WhiskPromptLibrary:
    """
    Library of proven prompts for Whisk-style generation
    Based on what works well with WAN + Qwen2.5-VL
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "category": ([
                    "character",
                    "environment", 
                    "action",
                    "cinematic",
                    "abstract"
                ], {"default": "character"}),
                "template": ("INT", {"default": 0, "min": 0, "max": 10}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt_template", "usage_guide")
    FUNCTION = "get_template"
    CATEGORY = "QwenWANBridge/Whisk"
    
    def get_template(self, category, template):
        
        templates = {
            "character": [
                "A {subject} with {features}, wearing {clothing}, {action} in {setting}, {mood} lighting, {style} style",
                "Portrait of {subject}, {expression}, {pose}, {background}, cinematic lighting, high detail",
                "Full body shot of {character}, {outfit}, {action}, {environment}, dynamic composition",
            ],
            "environment": [
                "Wide shot of {location}, {time_of_day}, {weather}, {atmosphere}, establishing shot",
                "{style} landscape featuring {elements}, {lighting}, {mood}, slow pan across scene",
                "Interior of {room}, {furniture}, {lighting}, {atmosphere}, static camera",
            ],
            "action": [
                "{subject} {action_verb} {direction}, {speed}, {style}, dynamic motion",
                "Tracking shot of {subject} {movement}, {setting}, smooth camera follow",
                "{character} transitions from {pose1} to {pose2}, {duration}, natural motion",
            ],
            "cinematic": [
                "{shot_type} of {subject}, {camera_movement}, {lighting_setup}, {color_grade}, filmic",
                "Dramatic {scene_type}, {focal_point}, depth of field, {atmosphere}, cinema quality",
                "{genre} style scene: {description}, {cinematography}, professional filming",
            ],
            "abstract": [
                "Abstract visualization of {concept}, {colors}, {movement_pattern}, {transformation}",
                "{shapes} morphing and {action}, {color_palette}, {rhythm}, mesmerizing",
                "Surreal {scene}, {elements}, {physics}, dreamlike quality, artistic",
            ]
        }
        
        # Get template for category
        category_templates = templates.get(category, templates["character"])
        template_idx = min(template, len(category_templates) - 1)
        selected = category_templates[template_idx]
        
        # Create usage guide
        guide = f"""Template Usage Guide
===================
Category: {category}
Template #{template}

Template:
{selected}

Fill in the {{placeholders}} with your specific details.

Example filled:
{self.example_for_category(category, template_idx)}

Tips for {category}:
{self.tips_for_category(category)}
"""
        
        return (selected, guide)
    
    def example_for_category(self, category, template_idx):
        examples = {
            "character": [
                "A young woman with long black hair, wearing a red dress, dancing gracefully in a ballroom, warm lighting, realistic style",
                "Portrait of an old wizard, wise expression, looking into distance, mystical forest background, cinematic lighting, high detail",
                "Full body shot of a robot, metallic armor, walking forward, futuristic city, dynamic composition",
            ],
            "environment": [
                "Wide shot of a mountain valley, golden hour, clear skies, serene atmosphere, establishing shot",
                "Fantasy landscape featuring floating islands, magical lighting, mystical mood, slow pan across scene",
                "Interior of a cozy library, wooden shelves, warm lamp light, peaceful atmosphere, static camera",
            ],
        }
        
        category_examples = examples.get(category, ["Example not available"])
        return category_examples[min(template_idx, len(category_examples) - 1)]
    
    def tips_for_category(self, category):
        tips = {
            "character": "• Include specific details about appearance\n• Describe the action clearly\n• Add emotion or mood",
            "environment": "• Set the time of day for lighting\n• Include weather or atmosphere\n• Specify camera movement",
            "action": "• Use clear action verbs\n• Specify speed and direction\n• Include camera behavior",
            "cinematic": "• Reference film techniques\n• Specify shot types and movements\n• Include lighting setup",
            "abstract": "• Embrace surreal descriptions\n• Focus on movement and transformation\n• Use artistic terminology",
        }
        return tips.get(category, "Be specific and descriptive")