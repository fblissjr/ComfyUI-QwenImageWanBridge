"""
Qwen Spatial Editor Node
Dedicated node for drawing spatial regions on images and generating spatial tokens
"""

import json
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class QwenSpatialEditor:
    """Interactive spatial editor for generating Qwen spatial tokens"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image to draw on
                "region_type": (["bounding_box", "object_reference", "polygon"], {
                    "default": "bounding_box"
                }),
                "region_label": ("STRING", {
                    "default": "object",
                    "placeholder": "Label for the region (e.g., 'car', 'tree', 'building')"
                }),
                "coordinates": ("STRING", {
                    "multiline": True,
                    "default": "100,100,300,300",
                    "placeholder": "Coordinates: x1,y1,x2,y2 for box or x1,y1 x2,y2 x3,y3 x4,y4 for polygon"
                }),
                "instruction_text": ("STRING", {
                    "multiline": True,
                    "default": "Edit this region",
                    "placeholder": "What should be done with this region?"
                }),
                "include_chat_format": ("BOOLEAN", {"default": True}),
                "include_vision_tokens": ("BOOLEAN", {"default": True}),
                "debug_mode": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "additional_regions": ("STRING", {
                    "multiline": True,
                    "placeholder": "JSON array of additional regions: [{\"type\":\"box\",\"coords\":\"x1,y1,x2,y2\",\"label\":\"object2\"}]"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("annotated_image", "spatial_tokens", "full_prompt", "coordinates_json", "debug_info")
    FUNCTION = "generate_spatial_tokens"
    CATEGORY = "Qwen/Spatial"
    OUTPUT_NODE = True
    
    def generate_spatial_tokens(self, image, region_type, region_label, coordinates, 
                               instruction_text, include_chat_format, include_vision_tokens,
                               debug_mode, additional_regions=""):
        """Generate spatial tokens and annotated image"""
        
        debug_info = []
        debug_info.append("=== SPATIAL EDITOR DEBUG ===")
        debug_info.append(f"Region type: {region_type}")
        debug_info.append(f"Region label: '{region_label}'")
        debug_info.append(f"Coordinates: '{coordinates}'")
        debug_info.append(f"Instruction: '{instruction_text}'")
        debug_info.append("")
        
        try:
            # Convert ComfyUI image tensor to PIL
            pil_image = self._tensor_to_pil(image)
            img_width, img_height = pil_image.size
            debug_info.append(f"Image dimensions: {img_width}x{img_height}")
            
            # Parse primary region
            primary_region = {
                "type": region_type,
                "label": region_label,
                "coords": coordinates.strip(),
                "instruction": instruction_text
            }
            
            regions = [primary_region]
            
            # Parse additional regions if provided
            if additional_regions.strip():
                try:
                    additional = json.loads(additional_regions)
                    regions.extend(additional)
                    debug_info.append(f"Added {len(additional)} additional regions")
                except json.JSONDecodeError as e:
                    debug_info.append(f"WARNING: Failed to parse additional regions: {e}")
            
            debug_info.append(f"Total regions to process: {len(regions)}")
            
            # Validate and normalize coordinates
            validated_regions = []
            for i, region in enumerate(regions):
                debug_info.append(f"\nProcessing region {i+1}:")
                validated_region = self._validate_region(region, img_width, img_height, debug_info)
                if validated_region:
                    validated_regions.append(validated_region)
            
            debug_info.append(f"\nValid regions: {len(validated_regions)}")
            
            # Generate annotated image
            annotated_image = self._draw_regions_on_image(pil_image, validated_regions, debug_info)
            
            # Generate spatial tokens
            spatial_tokens = self._generate_spatial_tokens(validated_regions, debug_info)
            
            # Generate full prompt
            full_prompt = self._generate_full_prompt(
                spatial_tokens, instruction_text, include_chat_format, 
                include_vision_tokens, debug_info
            )
            
            # Create coordinates JSON
            coordinates_json = json.dumps({
                "image_dimensions": {"width": img_width, "height": img_height},
                "regions": validated_regions
            }, indent=2)
            
            # Convert back to ComfyUI tensor
            output_image = self._pil_to_tensor(annotated_image)
            
            debug_text = "\n".join(debug_info) if debug_mode else "Debug mode disabled"
            
            return (
                output_image,
                spatial_tokens,
                full_prompt, 
                coordinates_json,
                debug_text
            )
            
        except Exception as e:
            logger.error(f"Error in spatial editor: {str(e)}")
            error_msg = f"ERROR: {str(e)}\n\nDebug info:\n" + "\n".join(debug_info)
            
            # Return original image on error
            return (
                image,
                f"Error: {str(e)}",
                f"Error: {str(e)}",
                json.dumps({"error": str(e)}),
                error_msg
            )
    
    def _tensor_to_pil(self, tensor):
        """Convert ComfyUI image tensor to PIL Image"""
        # ComfyUI images are typically (batch, height, width, channels)
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Take first image if batch
        
        # Convert to numpy and ensure proper range
        np_image = tensor.cpu().numpy()
        if np_image.max() <= 1.0:
            np_image = (np_image * 255).astype(np.uint8)
        else:
            np_image = np_image.astype(np.uint8)
        
        # Convert to PIL
        pil_image = Image.fromarray(np_image)
        return pil_image
    
    def _pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image).unsqueeze(0)  # Add batch dimension
        return tensor
    
    def _validate_region(self, region, img_width, img_height, debug_info):
        """Validate and normalize region coordinates"""
        region_type = region.get("type", "bounding_box")
        coords_str = region.get("coords", "")
        label = region.get("label", "object")
        
        debug_info.append(f"  Type: {region_type}")
        debug_info.append(f"  Label: '{label}'")
        debug_info.append(f"  Raw coords: '{coords_str}'")
        
        try:
            if region_type == "bounding_box":
                # Parse x1,y1,x2,y2
                coords = [float(c.strip()) for c in coords_str.split(',')]
                if len(coords) != 4:
                    debug_info.append(f"  ERROR: Expected 4 coordinates, got {len(coords)}")
                    return None
                
                x1, y1, x2, y2 = coords
                
                # Normalize if coordinates appear to be in 0-1 range
                if all(c <= 1.0 for c in coords):
                    x1, x2 = x1 * img_width, x2 * img_width
                    y1, y2 = y1 * img_height, y2 * img_height
                    debug_info.append(f"  Normalized coordinates from relative to absolute")
                
                # Ensure proper ordering
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Clamp to image bounds
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                
                validated_coords = [int(x1), int(y1), int(x2), int(y2)]
                debug_info.append(f"  Validated coords: {validated_coords}")
                
                return {
                    "type": region_type,
                    "label": label,
                    "coords": validated_coords,
                    "coords_str": f"{validated_coords[0]},{validated_coords[1]},{validated_coords[2]},{validated_coords[3]}",
                    "instruction": region.get("instruction", "")
                }
            
            elif region_type == "polygon":
                # Parse space-separated x,y pairs
                coord_pairs = coords_str.strip().split()
                if len(coord_pairs) < 3:
                    debug_info.append(f"  ERROR: Polygon needs at least 3 coordinate pairs, got {len(coord_pairs)}")
                    return None
                
                validated_pairs = []
                for pair in coord_pairs:
                    if ',' not in pair:
                        debug_info.append(f"  ERROR: Invalid coordinate pair: '{pair}'")
                        return None
                    
                    x_str, y_str = pair.split(',', 1)
                    x, y = float(x_str), float(y_str)
                    
                    # Normalize if needed
                    if x <= 1.0 and y <= 1.0:
                        x, y = x * img_width, y * img_height
                    
                    # Clamp to bounds
                    x = max(0, min(x, img_width))
                    y = max(0, min(y, img_height))
                    
                    validated_pairs.append([int(x), int(y)])
                
                debug_info.append(f"  Validated polygon: {len(validated_pairs)} points")
                
                return {
                    "type": region_type,
                    "label": label,
                    "coords": validated_pairs,
                    "coords_str": " ".join([f"{p[0]},{p[1]}" for p in validated_pairs]),
                    "instruction": region.get("instruction", "")
                }
            
            else:
                debug_info.append(f"  ERROR: Unsupported region type: {region_type}")
                return None
                
        except (ValueError, IndexError) as e:
            debug_info.append(f"  ERROR: Failed to parse coordinates: {e}")
            return None
    
    def _draw_regions_on_image(self, pil_image, regions, debug_info):
        """Draw regions on the image with annotations"""
        debug_info.append("\n=== DRAWING REGIONS ===")
        
        # Create a copy to draw on
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Define colors for different regions
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        try:
            # Try to load a font
            font_size = max(12, min(pil_image.width, pil_image.height) // 40)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        for i, region in enumerate(regions):
            color = colors[i % len(colors)]
            region_type = region["type"]
            label = region["label"]
            coords = region["coords"]
            
            debug_info.append(f"Drawing region {i+1}: {region_type} '{label}'")
            
            if region_type == "bounding_box":
                x1, y1, x2, y2 = coords
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label
                label_text = f"{i+1}. {label}"
                draw.text((x1, y1-20), label_text, fill=color, font=font)
                
                # Draw coordinates
                coord_text = f"({x1},{y1})-({x2},{y2})"
                draw.text((x1, y2+5), coord_text, fill=color, font=font)
                
            elif region_type == "polygon":
                # Draw polygon
                points = [tuple(p) for p in coords]
                if len(points) >= 3:
                    draw.polygon(points, outline=color, width=3)
                    
                    # Draw label at first point
                    label_text = f"{i+1}. {label}"
                    draw.text((points[0][0], points[0][1]-20), label_text, fill=color, font=font)
                    
                    # Draw point numbers
                    for j, point in enumerate(points):
                        draw.ellipse([point[0]-3, point[1]-3, point[0]+3, point[1]+3], fill=color)
                        draw.text((point[0]+5, point[1]+5), str(j+1), fill=color, font=font)
        
        debug_info.append(f"Drew {len(regions)} regions on image")
        return annotated
    
    def _generate_spatial_tokens(self, regions, debug_info):
        """Generate spatial tokens from regions"""
        debug_info.append("\n=== GENERATING SPATIAL TOKENS ===")
        
        tokens = []
        
        for i, region in enumerate(regions):
            region_type = region["type"]
            label = region["label"] 
            coords_str = region["coords_str"]
            
            debug_info.append(f"Region {i+1}: {region_type} '{label}'")
            
            if region_type == "bounding_box":
                # Generate: <|object_ref_start|>label<|object_ref_end|> at <|box_start|>coords<|box_end|>
                token = f"<|object_ref_start|>{label}<|object_ref_end|> at <|box_start|>{coords_str}<|box_end|>"
                tokens.append(token)
                debug_info.append(f"  Generated: {token}")
                
            elif region_type == "polygon":
                # Generate: <|object_ref_start|>label<|object_ref_end|> outlined by <|quad_start|>coords<|quad_end|>
                token = f"<|object_ref_start|>{label}<|object_ref_end|> outlined by <|quad_start|>{coords_str}<|quad_end|>"
                tokens.append(token)
                debug_info.append(f"  Generated: {token}")
            
            elif region_type == "object_reference":
                # Generate: <|object_ref_start|>label<|object_ref_end|>
                token = f"<|object_ref_start|>{label}<|object_ref_end|>"
                tokens.append(token)
                debug_info.append(f"  Generated: {token}")
        
        combined_tokens = " and ".join(tokens) if len(tokens) > 1 else (tokens[0] if tokens else "")
        debug_info.append(f"Combined spatial tokens: {combined_tokens}")
        
        return combined_tokens
    
    def _generate_full_prompt(self, spatial_tokens, instruction_text, include_chat_format, 
                             include_vision_tokens, debug_info):
        """Generate complete prompt with spatial tokens"""
        debug_info.append("\n=== GENERATING FULL PROMPT ===")
        
        # Base instruction
        if spatial_tokens:
            base_text = f"{instruction_text} {spatial_tokens}"
        else:
            base_text = instruction_text
        
        # Add vision tokens if requested
        if include_vision_tokens:
            base_text = f"{base_text} in this image: <|vision_start|><|image_pad|><|vision_end|>"
            debug_info.append("Added vision tokens")
        
        # Add chat format if requested
        if include_chat_format:
            full_prompt = f"<|im_start|>user\n{base_text}<|im_end|>"
            debug_info.append("Added chat format")
        else:
            full_prompt = base_text
        
        debug_info.append(f"Final prompt length: {len(full_prompt)} characters")
        return full_prompt


# Register the node
NODE_CLASS_MAPPINGS = {
    "QwenSpatialEditor": QwenSpatialEditor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenSpatialEditor": "Qwen Spatial Editor"
}