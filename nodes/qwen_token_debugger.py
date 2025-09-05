"""
Qwen Token Debugger Node
Provides detailed analysis of token sequences and special token usage
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
import os


class QwenTokenDebugger:
    """Debug and analyze Qwen tokenizer sequences"""
    
    # Token definitions from our analysis
    QWEN_TOKENS = {
        "VISION": {
            "<|vision_start|>": 151652,
            "<|vision_end|>": 151653,
            "<|image_pad|>": 151655,
            "<|video_pad|>": 151656,
            "<|vision_pad|>": 151654
        },
        "SPATIAL": {
            "<|object_ref_start|>": 151646,
            "<|object_ref_end|>": 151647,
            "<|box_start|>": 151648,
            "<|box_end|>": 151649,
            "<|quad_start|>": 151650,
            "<|quad_end|>": 151651
        },
        "CHAT": {
            "<|im_start|>": 151644,
            "<|im_end|>": 151645
        },
        "CONTROL": {
            "<|endoftext|>": 151643
        },
        "CODE": {
            "<|fim_prefix|>": 151659,
            "<|fim_middle|>": 151660,
            "<|fim_suffix|>": 151661,
            "<|fim_pad|>": 151662,
            "<|repo_name|>": 151663,
            "<|file_sep|>": 151664
        },
        "TOOL": {
            "<tool_call>": 151657,
            "</tool_call>": 151658
        }
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "placeholder": "Enter text with Qwen tokens to analyze...",
                    "default": "Describe this image: <|vision_start|><|image_pad|><|vision_end|>"
                }),
                "analysis_mode": (["full", "tokens_only", "sequences_only", "errors_only"], {
                    "default": "full"
                }),
                "include_templates": ("BOOLEAN", {"default": True}),
                "validate_coordinates": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "context_text": ("STRING", {
                    "multiline": True,
                    "placeholder": "Additional context for analysis (optional)"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("analysis_report", "validated_text", "template_suggestions", 
                   "total_special_tokens", "vision_tokens", "spatial_tokens")
    FUNCTION = "analyze_tokens"
    CATEGORY = "Qwen/Debug"
    
    def __init__(self):
        self.all_tokens = {}
        for category in self.QWEN_TOKENS.values():
            self.all_tokens.update(category)
    
    def analyze_tokens(self, text: str, analysis_mode: str, include_templates: bool, 
                      validate_coordinates: bool, context_text: str = ""):
        """Analyze token sequences in the input text"""
        
        # Combine text and context
        full_text = f"{context_text}\n{text}" if context_text else text
        
        # Perform analysis
        analysis = self._analyze_text(full_text)
        validation = self._validate_sequences(analysis["sequences"]) if validate_coordinates else []
        templates = self._generate_templates(full_text) if include_templates else []
        
        # Format output based on mode
        if analysis_mode == "full":
            report = self._format_full_report(analysis, validation, templates)
        elif analysis_mode == "tokens_only":
            report = self._format_token_report(analysis)
        elif analysis_mode == "sequences_only":
            report = self._format_sequence_report(analysis)
        elif analysis_mode == "errors_only":
            report = self._format_error_report(validation)
        else:
            report = self._format_full_report(analysis, validation, templates)
        
        # Create validated/corrected text
        validated_text = self._create_validated_text(full_text, validation)
        
        # Template suggestions
        template_text = "\n".join(templates) if templates else "No template suggestions"
        
        return (
            report,
            validated_text,
            template_text,
            analysis["special_token_count"],
            analysis["vision_token_count"],
            analysis["spatial_token_count"]
        )
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for token usage patterns"""
        analysis = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "special_token_count": 0,
            "vision_token_count": 0,
            "spatial_token_count": 0,
            "chat_token_count": 0,
            "code_token_count": 0,
            "tool_token_count": 0,
            "token_breakdown": {},
            "sequences": [],
            "pattern_matches": []
        }
        
        # Count each special token
        for category_name, category_tokens in self.QWEN_TOKENS.items():
            category_count = 0
            for token, token_id in category_tokens.items():
                count = len(re.findall(re.escape(token), text))
                if count > 0:
                    analysis["token_breakdown"][token] = count
                    analysis["special_token_count"] += count
                    category_count += count
            
            # Update category counts
            if category_name == "VISION":
                analysis["vision_token_count"] = category_count
            elif category_name == "SPATIAL":
                analysis["spatial_token_count"] = category_count
            elif category_name == "CHAT":
                analysis["chat_token_count"] = category_count
            elif category_name == "CODE":
                analysis["code_token_count"] = category_count
            elif category_name == "TOOL":
                analysis["tool_token_count"] = category_count
        
        # Find sequences
        analysis["sequences"] = self._find_sequences(text)
        
        # Find common patterns
        analysis["pattern_matches"] = self._find_patterns(text)
        
        return analysis
    
    def _find_sequences(self, text: str) -> List[Dict[str, Any]]:
        """Find structured token sequences"""
        sequences = []
        
        # Vision processing sequences
        vision_pattern = r'<\|vision_start\|>(.*?)<\|vision_end\|>'
        for match in re.finditer(vision_pattern, text, re.DOTALL):
            sequences.append({
                "type": "vision_processing",
                "content": match.group(0),
                "inner_content": match.group(1),
                "start": match.start(),
                "end": match.end(),
                "length": len(match.group(0))
            })
        
        # Spatial reference sequences
        box_pattern = r'<\|box_start\|>(.*?)<\|box_end\|>'
        for match in re.finditer(box_pattern, text):
            sequences.append({
                "type": "bounding_box",
                "content": match.group(0),
                "coordinates": match.group(1).strip(),
                "start": match.start(),
                "end": match.end(),
                "length": len(match.group(0))
            })
        
        # Quad/polygon sequences
        quad_pattern = r'<\|quad_start\|>(.*?)<\|quad_end\|>'
        for match in re.finditer(quad_pattern, text):
            sequences.append({
                "type": "polygon",
                "content": match.group(0),
                "coordinates": match.group(1).strip(),
                "start": match.start(),
                "end": match.end(),
                "length": len(match.group(0))
            })
        
        # Object reference sequences
        obj_pattern = r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>'
        for match in re.finditer(obj_pattern, text):
            sequences.append({
                "type": "object_reference",
                "content": match.group(0),
                "object_name": match.group(1).strip(),
                "start": match.start(),
                "end": match.end(),
                "length": len(match.group(0))
            })
        
        # Chat sequences
        chat_pattern = r'<\|im_start\|>(.*?)<\|im_end\|>'
        for match in re.finditer(chat_pattern, text, re.DOTALL):
            sequences.append({
                "type": "chat_message",
                "content": match.group(0),
                "message_content": match.group(1).strip(),
                "start": match.start(),
                "end": match.end(),
                "length": len(match.group(0))
            })
        
        return sorted(sequences, key=lambda x: x["start"])
    
    def _find_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Find common usage patterns"""
        patterns = []
        
        # Chat + Vision pattern
        chat_vision_pattern = r'<\|im_start\|>.*?<\|vision_start\|>.*?<\|vision_end\|>.*?<\|im_end\|>'
        if re.search(chat_vision_pattern, text, re.DOTALL):
            patterns.append({
                "name": "chat_with_vision",
                "description": "Chat message containing vision processing",
                "complexity": "medium"
            })
        
        # Spatial editing pattern
        spatial_edit_pattern = r'<\|object_ref_start\|>.*?<\|object_ref_end\|>.*?<\|box_start\|>.*?<\|box_end\|>'
        if re.search(spatial_edit_pattern, text):
            patterns.append({
                "name": "spatial_object_editing",
                "description": "Object reference with spatial coordinates",
                "complexity": "high"
            })
        
        # Multi-modal instruction pattern
        multimodal_pattern = r'<\|vision_start\|>.*?<\|vision_end\|>.*?<\|box_start\|>.*?<\|box_end\|>'
        if re.search(multimodal_pattern, text):
            patterns.append({
                "name": "multimodal_instruction",
                "description": "Vision processing with spatial references",
                "complexity": "high"
            })
        
        return patterns
    
    def _validate_sequences(self, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate token sequences for common errors"""
        errors = []
        
        for seq in sequences:
            if seq["type"] == "bounding_box":
                coord_errors = self._validate_coordinates(seq["coordinates"], "box")
                if coord_errors:
                    errors.extend([{
                        "sequence": seq,
                        "error_type": "coordinate_validation",
                        "errors": coord_errors
                    }])
            
            elif seq["type"] == "polygon":
                coord_errors = self._validate_coordinates(seq["coordinates"], "polygon")
                if coord_errors:
                    errors.extend([{
                        "sequence": seq,
                        "error_type": "coordinate_validation", 
                        "errors": coord_errors
                    }])
            
            elif seq["type"] == "vision_processing":
                if not seq["inner_content"].strip():
                    errors.append({
                        "sequence": seq,
                        "error_type": "empty_vision_content",
                        "errors": ["Vision sequence has no content between start/end tokens"]
                    })
        
        return errors
    
    def _validate_coordinates(self, coord_str: str, coord_type: str) -> List[str]:
        """Validate coordinate strings"""
        errors = []
        
        if coord_type == "box":
            # Expecting x1,y1,x2,y2
            coords = [c.strip() for c in coord_str.split(',')]
            if len(coords) != 4:
                errors.append(f"Box coordinates should have 4 values, got {len(coords)}")
            else:
                for i, coord in enumerate(coords):
                    try:
                        float(coord)
                    except ValueError:
                        errors.append(f"Invalid coordinate at position {i}: '{coord}'")
        
        elif coord_type == "polygon":
            # Expecting pairs of x,y coordinates
            coords = coord_str.strip().split()
            if len(coords) % 2 != 0:
                errors.append(f"Polygon coordinates should be in x,y pairs, got {len(coords)} values")
            else:
                for i, coord_pair in enumerate(coords):
                    if ',' not in coord_pair:
                        errors.append(f"Coordinate pair {i} missing comma: '{coord_pair}'")
                    else:
                        x, y = coord_pair.split(',')
                        try:
                            float(x)
                            float(y)
                        except ValueError:
                            errors.append(f"Invalid coordinate pair {i}: '{coord_pair}'")
        
        return errors
    
    def _generate_templates(self, text: str) -> List[str]:
        """Generate template suggestions based on analysis"""
        templates = []
        
        # Basic templates
        if "<|vision_start|>" not in text:
            templates.append("Basic Vision: <|vision_start|><|image_pad|><|vision_end|>")
        
        if "<|im_start|>" not in text and ("<|vision_start|>" in text or "image" in text.lower()):
            templates.append("Chat + Vision: <|im_start|>user\\nAnalyze this image: <|vision_start|><|image_pad|><|vision_end|>\\nWhat do you see?<|im_end|>")
        
        if "<|box_start|>" not in text and ("edit" in text.lower() or "region" in text.lower()):
            templates.append("Spatial Edit: Edit the <|object_ref_start|>object<|object_ref_end|> at <|box_start|>100,100,200,200<|box_end|>")
        
        if "<|object_ref_start|>" not in text and "<|box_start|>" in text:
            templates.append("Object Reference: The <|object_ref_start|>target object<|object_ref_end|> is located at existing coordinates")
        
        # Advanced templates based on detected patterns
        has_vision = "<|vision_start|>" in text
        has_spatial = "<|box_start|>" in text or "<|quad_start|>" in text
        has_chat = "<|im_start|>" in text
        
        if has_vision and has_spatial and not has_chat:
            templates.append("Complete Multi-modal: <|im_start|>user\\nEdit the region <|box_start|>x1,y1,x2,y2<|box_end|> in this image: <|vision_start|><|image_pad|><|vision_end|>\\nYour instruction here<|im_end|>")
        
        return templates
    
    def _create_validated_text(self, text: str, validation_errors: List[Dict[str, Any]]) -> str:
        """Create corrected version of the text"""
        if not validation_errors:
            return text
        
        corrected = text
        corrections_made = []
        
        for error_info in validation_errors:
            if error_info["error_type"] == "coordinate_validation":
                seq = error_info["sequence"]
                if seq["type"] == "bounding_box":
                    # Try to fix coordinate format
                    coords = seq["coordinates"].split(',')
                    if len(coords) == 4:
                        try:
                            fixed_coords = ','.join([str(int(float(c.strip()))) for c in coords])
                            old_content = seq["content"]
                            new_content = f"<|box_start|>{fixed_coords}<|box_end|>"
                            corrected = corrected.replace(old_content, new_content, 1)
                            corrections_made.append(f"Fixed box coordinates: {seq['coordinates']} -> {fixed_coords}")
                        except ValueError:
                            corrections_made.append(f"Could not auto-fix coordinates: {seq['coordinates']}")
        
        if corrections_made:
            corrected += "\n\n# Auto-corrections made:\n" + "\n".join(f"- {c}" for c in corrections_made)
        
        return corrected
    
    def _format_full_report(self, analysis: Dict[str, Any], validation: List[Dict[str, Any]], 
                           templates: List[str]) -> str:
        """Format comprehensive analysis report"""
        report = []
        
        report.append("=== QWEN TOKEN ANALYSIS REPORT ===\\n")
        
        # Basic stats
        report.append(f"Text Length: {analysis['text_length']} chars")
        report.append(f"Word Count: {analysis['word_count']} words")
        report.append(f"Special Tokens: {analysis['special_token_count']}\\n")
        
        # Token breakdown by category
        report.append("Token Breakdown:")
        report.append(f"  Vision: {analysis['vision_token_count']}")
        report.append(f"  Spatial: {analysis['spatial_token_count']}")
        report.append(f"  Chat: {analysis['chat_token_count']}")
        report.append(f"  Code: {analysis['code_token_count']}")
        report.append(f"  Tool: {analysis['tool_token_count']}\\n")
        
        # Individual token counts
        if analysis['token_breakdown']:
            report.append("Individual Token Counts:")
            for token, count in analysis['token_breakdown'].items():
                report.append(f"  {token}: {count}")
            report.append("")
        
        # Sequences found
        if analysis['sequences']:
            report.append(f"Sequences Found ({len(analysis['sequences'])}):")
            for i, seq in enumerate(analysis['sequences'], 1):
                report.append(f"  {i}. {seq['type']}: {seq['content'][:50]}...")
            report.append("")
        
        # Patterns detected
        if analysis['pattern_matches']:
            report.append("Patterns Detected:")
            for pattern in analysis['pattern_matches']:
                report.append(f"  - {pattern['name']}: {pattern['description']} ({pattern['complexity']} complexity)")
            report.append("")
        
        # Validation errors
        if validation:
            report.append(f"Validation Errors ({len(validation)}):")
            for error in validation:
                report.append(f"  - {error['error_type']}: {error['errors']}")
            report.append("")
        
        # Template suggestions
        if templates:
            report.append("Template Suggestions:")
            for template in templates:
                report.append(f"  - {template}")
        
        return "\\n".join(report)
    
    def _format_token_report(self, analysis: Dict[str, Any]) -> str:
        """Format token-focused report"""
        report = []
        report.append("=== TOKEN USAGE REPORT ===\\n")
        report.append(f"Total Special Tokens: {analysis['special_token_count']}")
        
        for token, count in analysis['token_breakdown'].items():
            report.append(f"{token}: {count}")
        
        return "\\n".join(report)
    
    def _format_sequence_report(self, analysis: Dict[str, Any]) -> str:
        """Format sequence-focused report"""
        report = []
        report.append("=== SEQUENCE ANALYSIS ===\\n")
        
        for seq in analysis['sequences']:
            report.append(f"Type: {seq['type']}")
            report.append(f"Content: {seq['content']}")
            report.append(f"Position: {seq['start']}-{seq['end']} ({seq['length']} chars)")
            report.append("")
        
        return "\\n".join(report)
    
    def _format_error_report(self, validation: List[Dict[str, Any]]) -> str:
        """Format error-focused report"""
        if not validation:
            return "No validation errors found."
        
        report = []
        report.append("=== VALIDATION ERRORS ===\\n")
        
        for error in validation:
            report.append(f"Error Type: {error['error_type']}")
            report.append(f"Sequence: {error['sequence']['type']}")
            report.append(f"Errors: {', '.join(error['errors'])}")
            report.append("")
        
        return "\\n".join(report)


# Register the node
NODE_CLASS_MAPPINGS = {
    "QwenTokenDebugger": QwenTokenDebugger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTokenDebugger": "Qwen Token Debugger"
}