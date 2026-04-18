"""
Standalone Qwen Token Analyzer Node
Dedicated node for token analysis with proper data flow
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class QwenTokenAnalyzerStandalone:
    """Standalone token analyzer with visual output"""
    
    # Token definitions
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
                "input_text": ("STRING", {
                    "multiline": True,
                    "placeholder": "Enter text with Qwen tokens to analyze...",
                    "default": "Describe this image: <|vision_start|><|image_pad|><|vision_end|>"
                }),
                "debug_mode": ("BOOLEAN", {"default": True}),
                "show_token_ids": ("BOOLEAN", {"default": True}),
                "validate_coordinates": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("analysis_json", "token_breakdown", "sequences_found", "debug_info", 
                   "total_special_tokens", "estimated_total_tokens")
    FUNCTION = "analyze_tokens"
    CATEGORY = "Qwen/Analysis"
    OUTPUT_NODE = True  # This makes it show outputs in UI
    
    def __init__(self):
        self.all_tokens = {}
        for category in self.QWEN_TOKENS.values():
            self.all_tokens.update(category)
        logger.info(f"QwenTokenAnalyzer initialized with {len(self.all_tokens)} special tokens")
    
    def analyze_tokens(self, input_text: str, debug_mode: bool, show_token_ids: bool, 
                      validate_coordinates: bool):
        """Analyze token sequences with full debugging"""
        
        logger.info(f"Starting token analysis for text length: {len(input_text)}")
        
        debug_info = []
        debug_info.append(f"=== TOKEN ANALYSIS DEBUG ===")
        debug_info.append(f"Input length: {len(input_text)} characters")
        debug_info.append(f"Debug mode: {debug_mode}")
        debug_info.append(f"Show token IDs: {show_token_ids}")
        debug_info.append(f"Validate coordinates: {validate_coordinates}")
        debug_info.append("")
        
        try:
            # 1. Basic text analysis
            word_count = len(input_text.split())
            estimated_tokens = word_count + input_text.count(' ')
            debug_info.append(f"Estimated total tokens: {estimated_tokens}")
            
            # 2. Find all special tokens
            special_tokens_found = {}
            total_special_tokens = 0
            
            debug_info.append("\n=== SPECIAL TOKEN SEARCH ===")
            for category_name, category_tokens in self.QWEN_TOKENS.items():
                debug_info.append(f"\nSearching {category_name} tokens:")
                category_count = 0
                
                for token, token_id in category_tokens.items():
                    count = len(re.findall(re.escape(token), input_text))
                    if count > 0:
                        special_tokens_found[token] = {
                            "count": count,
                            "id": token_id,
                            "category": category_name
                        }
                        total_special_tokens += count
                        category_count += count
                        debug_info.append(f"  ✓ {token} (ID: {token_id}): {count} occurrences")
                    elif debug_mode:
                        debug_info.append(f"  - {token}: not found")
                
                if category_count > 0:
                    debug_info.append(f"  {category_name} total: {category_count}")
            
            debug_info.append(f"\nTotal special tokens found: {total_special_tokens}")
            
            # 3. Find token sequences
            debug_info.append("\n=== SEQUENCE ANALYSIS ===")
            sequences = self._find_sequences(input_text, debug_info)
            
            # 4. Validate coordinates if requested
            validation_errors = []
            if validate_coordinates:
                debug_info.append("\n=== COORDINATE VALIDATION ===")
                validation_errors = self._validate_coordinates(input_text, debug_info)
            
            # 5. Create comprehensive analysis JSON
            analysis = {
                "input_length": len(input_text),
                "estimated_tokens": estimated_tokens,
                "special_tokens_count": total_special_tokens,
                "special_tokens": special_tokens_found,
                "sequences": sequences,
                "validation_errors": validation_errors,
                "token_efficiency": total_special_tokens / max(estimated_tokens, 1),
                "analysis_timestamp": self._get_timestamp()
            }
            
            # 6. Create formatted outputs
            analysis_json = json.dumps(analysis, indent=2)
            
            # Token breakdown text
            token_breakdown = self._format_token_breakdown(special_tokens_found, show_token_ids)
            
            # Sequences text
            sequences_text = self._format_sequences(sequences)
            
            # Debug info text
            debug_text = "\n".join(debug_info) if debug_mode else "Debug mode disabled"
            
            logger.info(f"Analysis completed successfully. Found {len(sequences)} sequences")
            
            return (
                analysis_json,
                token_breakdown, 
                sequences_text,
                debug_text,
                total_special_tokens,
                estimated_tokens
            )
            
        except Exception as e:
            logger.error(f"Error in token analysis: {str(e)}")
            error_info = f"ERROR in token analysis: {str(e)}\n\nDebug info:\n" + "\n".join(debug_info)
            return (
                json.dumps({"error": str(e)}),
                f"Error: {str(e)}",
                "Error occurred",
                error_info,
                0,
                0
            )
    
    def _find_sequences(self, text: str, debug_info: List[str]) -> List[Dict[str, Any]]:
        """Find structured token sequences with debugging"""
        sequences = []
        
        # Vision processing sequences
        debug_info.append("Searching for vision sequences...")
        vision_pattern = r'<\|vision_start\|>(.*?)<\|vision_end\|>'
        for match in re.finditer(vision_pattern, text, re.DOTALL):
            seq = {
                "type": "vision_processing",
                "content": match.group(0),
                "inner_content": match.group(1).strip(),
                "start": match.start(),
                "end": match.end(),
                "length": len(match.group(0))
            }
            sequences.append(seq)
            debug_info.append(f"  Vision sequence: pos {seq['start']}-{seq['end']}, inner: '{seq['inner_content'][:50]}...'")
        
        # Spatial reference sequences
        debug_info.append("Searching for spatial sequences...")
        
        # Bounding boxes
        box_pattern = r'<\|box_start\|>(.*?)<\|box_end\|>'
        for match in re.finditer(box_pattern, text):
            seq = {
                "type": "bounding_box",
                "content": match.group(0),
                "coordinates": match.group(1).strip(),
                "start": match.start(),
                "end": match.end(),
                "length": len(match.group(0))
            }
            sequences.append(seq)
            debug_info.append(f"  Bounding box: pos {seq['start']}-{seq['end']}, coords: '{seq['coordinates']}'")
        
        # Object references
        obj_pattern = r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>'
        for match in re.finditer(obj_pattern, text):
            seq = {
                "type": "object_reference",
                "content": match.group(0),
                "object_name": match.group(1).strip(),
                "start": match.start(),
                "end": match.end(),
                "length": len(match.group(0))
            }
            sequences.append(seq)
            debug_info.append(f"  Object reference: pos {seq['start']}-{seq['end']}, object: '{seq['object_name']}'")
        
        # Chat sequences
        debug_info.append("Searching for chat sequences...")
        chat_pattern = r'<\|im_start\|>(.*?)<\|im_end\|>'
        for match in re.finditer(chat_pattern, text, re.DOTALL):
            seq = {
                "type": "chat_message",
                "content": match.group(0),
                "message_content": match.group(1).strip(),
                "start": match.start(),
                "end": match.end(),
                "length": len(match.group(0))
            }
            sequences.append(seq)
            debug_info.append(f"  Chat message: pos {seq['start']}-{seq['end']}, content: '{seq['message_content'][:50]}...'")
        
        debug_info.append(f"Total sequences found: {len(sequences)}")
        return sorted(sequences, key=lambda x: x["start"])
    
    def _validate_coordinates(self, text: str, debug_info: List[str]) -> List[Dict[str, Any]]:
        """Validate coordinate sequences with debugging"""
        errors = []
        
        # Validate box coordinates
        box_pattern = r'<\|box_start\|>(.*?)<\|box_end\|>'
        box_matches = list(re.finditer(box_pattern, text))
        debug_info.append(f"Validating {len(box_matches)} bounding box coordinate sequences...")
        
        for match in box_matches:
            coords_str = match.group(1).strip()
            debug_info.append(f"  Checking box coords: '{coords_str}'")
            
            coords = [c.strip() for c in coords_str.split(',')]
            if len(coords) != 4:
                error = {
                    "type": "coordinate_count",
                    "sequence": match.group(0),
                    "expected": 4,
                    "found": len(coords),
                    "position": match.start(),
                    "message": f"Box coordinates should have 4 values, got {len(coords)}"
                }
                errors.append(error)
                debug_info.append(f"    ERROR: Expected 4 coordinates, got {len(coords)}")
            else:
                # Validate each coordinate
                valid_coords = []
                for i, coord in enumerate(coords):
                    try:
                        val = float(coord)
                        valid_coords.append(val)
                        debug_info.append(f"    Coord {i}: {coord} -> {val} ✓")
                    except ValueError:
                        error = {
                            "type": "coordinate_format",
                            "sequence": match.group(0),
                            "coordinate_index": i,
                            "coordinate_value": coord,
                            "position": match.start(),
                            "message": f"Invalid coordinate at position {i}: '{coord}'"
                        }
                        errors.append(error)
                        debug_info.append(f"    ERROR at coord {i}: '{coord}' is not a valid number")
                        break
                
                # Check coordinate logic (x2 > x1, y2 > y1)
                if len(valid_coords) == 4:
                    x1, y1, x2, y2 = valid_coords
                    if x2 <= x1 or y2 <= y1:
                        error = {
                            "type": "coordinate_logic",
                            "sequence": match.group(0),
                            "coordinates": valid_coords,
                            "position": match.start(),
                            "message": f"Box coordinates invalid: x2({x2}) must > x1({x1}) and y2({y2}) must > y1({y1})"
                        }
                        errors.append(error)
                        debug_info.append(f"    ERROR: Invalid box logic - x2({x2}) > x1({x1})? {x2>x1}, y2({y2}) > y1({y1})? {y2>y1}")
                    else:
                        debug_info.append(f"    Box coordinates valid: ({x1},{y1}) to ({x2},{y2})")
        
        debug_info.append(f"Coordinate validation complete. {len(errors)} errors found.")
        return errors
    
    def _format_token_breakdown(self, tokens: Dict[str, Any], show_ids: bool) -> str:
        """Format token breakdown for display"""
        if not tokens:
            return "No special tokens found."
        
        lines = ["=== TOKEN BREAKDOWN ===\n"]
        
        # Group by category
        by_category = {}
        for token, info in tokens.items():
            category = info["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((token, info))
        
        for category, token_list in by_category.items():
            lines.append(f"{category}:")
            total_count = 0
            for token, info in sorted(token_list, key=lambda x: x[1]["id"]):
                count = info["count"]
                total_count += count
                if show_ids:
                    lines.append(f"  {token} (ID: {info['id']}): {count}")
                else:
                    lines.append(f"  {token}: {count}")
            lines.append(f"  {category} Total: {total_count}\n")
        
        return "\n".join(lines)
    
    def _format_sequences(self, sequences: List[Dict[str, Any]]) -> str:
        """Format sequences for display"""
        if not sequences:
            return "No token sequences found."
        
        lines = ["=== TOKEN SEQUENCES ===\n"]
        
        for i, seq in enumerate(sequences, 1):
            lines.append(f"{i}. {seq['type'].upper()}")
            lines.append(f"   Position: {seq['start']}-{seq['end']} ({seq['length']} chars)")
            lines.append(f"   Content: {seq['content']}")
            
            if seq['type'] == 'vision_processing' and seq['inner_content']:
                lines.append(f"   Inner: {seq['inner_content']}")
            elif seq['type'] == 'bounding_box':
                lines.append(f"   Coordinates: {seq['coordinates']}")
            elif seq['type'] == 'object_reference':
                lines.append(f"   Object: {seq['object_name']}")
            elif seq['type'] == 'chat_message':
                lines.append(f"   Message: {seq['message_content'][:100]}...")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


# Register the node
NODE_CLASS_MAPPINGS = {
    "QwenTokenAnalyzerStandalone": QwenTokenAnalyzerStandalone
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTokenAnalyzerStandalone": "Qwen Token Analyzer"
}