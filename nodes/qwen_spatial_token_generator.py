"""
Qwen Spatial Token Generator
Pure spatial token generation without templates or assumptions
"""

import json
from PIL import Image, ImageDraw
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class QwenSpatialTokenGenerator:
    """Generate spatial tokens from image coordinates and labels"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "spatial_type": (["object_reference", "bounding_box", "polygon"], {
                    "default": "bounding_box"
                }),
                "label": ("STRING", {
                    "default": "object",
                    "placeholder": "Label like 'woman', 'green frog', 'red car'"
                }),
                "coordinates": ("STRING", {
                    "multiline": True,
                    "default": "100,100,300,300",
                    "placeholder": "Box: x1,y1,x2,y2 | Polygon: x1,y1 x2,y2 x3,y3 x4,y4"
                }),
                "normalize_coords": ("BOOLEAN", {"default": True}),
                "debug_mode": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "additional_regions": ("STRING", {
                    "multiline": True,
                    "placeholder": "JSON: [{\"type\":\"bounding_box\",\"label\":\"car\",\"coords\":\"50,50,150,150\"}]"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("annotated_image", "spatial_tokens", "debug_info")
    FUNCTION = "generate_tokens"
    CATEGORY = "Qwen/Spatial"
    OUTPUT_NODE = True
    
    def generate_tokens(self, image, spatial_type, label, coordinates, normalize_coords, 
                       debug_mode, additional_regions=""):
        """Generate spatial tokens from coordinates"""
        
        debug_info = []
        debug_info.append("=== SPATIAL TOKEN GENERATOR ===")
        
        try:
            # Convert tensor to PIL
            pil_image = self._tensor_to_pil(image)
            img_width, img_height = pil_image.size
            debug_info.append(f"Image: {img_width}x{img_height}")
            
            # Parse regions
            regions = [{"type": spatial_type, "label": label, "coords": coordinates}]
            
            if additional_regions.strip():
                try:
                    additional = json.loads(additional_regions)
                    regions.extend(additional)
                    debug_info.append(f"Added {len(additional)} additional regions")
                except json.JSONDecodeError as e:
                    debug_info.append(f"WARNING: Invalid additional regions JSON: {e}")
            
            # Process each region
            tokens = []
            annotated_image = pil_image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            for i, region in enumerate(regions):
                debug_info.append(f"\nRegion {i+1}: {region['type']} '{region['label']}'")
                
                try:
                    if region['type'] == 'bounding_box':
                        token = self._process_bounding_box(region, img_width, img_height, 
                                                         normalize_coords, draw, debug_info)
                    elif region['type'] == 'polygon':
                        token = self._process_polygon(region, img_width, img_height, 
                                                    normalize_coords, draw, debug_info)
                    elif region['type'] == 'object_reference':
                        token = self._process_object_reference(region, debug_info)
                    else:
                        debug_info.append(f"  ERROR: Unknown type '{region['type']}'")
                        continue
                    
                    if token:
                        tokens.append(token)
                        debug_info.append(f"  Generated: {token}")
                    
                except Exception as e:
                    debug_info.append(f"  ERROR: {str(e)}")
                    continue
            
            # Combine tokens
            spatial_tokens = " ".join(tokens) if tokens else ""
            debug_info.append(f"\nFinal tokens: {spatial_tokens}")
            
            # Convert back to tensor
            output_image = self._pil_to_tensor(annotated_image)
            debug_text = "\n".join(debug_info) if debug_mode else f"Generated {len(tokens)} spatial tokens"
            
            return (output_image, spatial_tokens, debug_text)
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            debug_info.append(error_msg)
            return (image, "", "\n".join(debug_info))
    
    def _process_bounding_box(self, region, img_width, img_height, normalize_coords, draw, debug_info):
        """Process bounding box coordinates"""
        coords_str = region['coords'].strip()
        label = region['label']
        
        # Parse x1,y1,x2,y2
        coords = [float(c.strip()) for c in coords_str.split(',')]
        if len(coords) != 4:
            raise ValueError(f"Bounding box needs 4 coordinates, got {len(coords)}")
        
        x1, y1, x2, y2 = coords
        
        # Normalize if needed
        if normalize_coords and all(c <= 1.0 for c in coords):
            x1, x2 = x1 * img_width, x2 * img_width
            y1, y2 = y1 * img_height, y2 * img_height
        
        # Ensure proper bounds
        x1, x2 = max(0, min(x1, x2)), min(img_width, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(img_height, max(y1, y2))
        
        # Draw annotation
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1-15), label, fill="red")
        
        # Generate normalized coordinates for token
        norm_x1, norm_y1 = x1 / img_width, y1 / img_height  
        norm_x2, norm_y2 = x2 / img_width, y2 / img_height
        
        debug_info.append(f"  Box coords: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        debug_info.append(f"  Normalized: ({norm_x1:.3f},{norm_y1:.3f},{norm_x2:.3f},{norm_y2:.3f})")
        
        return f"<|object_ref_start|>{label}<|object_ref_end|> at <|box_start|>{norm_x1:.3f},{norm_y1:.3f},{norm_x2:.3f},{norm_y2:.3f}<|box_end|>"
    
    def _process_polygon(self, region, img_width, img_height, normalize_coords, draw, debug_info):
        """Process polygon coordinates"""
        coords_str = region['coords'].strip()
        label = region['label']
        
        # Parse space-separated x,y pairs
        coord_pairs = coords_str.split()
        if len(coord_pairs) < 3:
            raise ValueError(f"Polygon needs at least 3 points, got {len(coord_pairs)}")
        
        points = []
        for pair in coord_pairs:
            if ',' not in pair:
                raise ValueError(f"Invalid coordinate pair: '{pair}'")
            x_str, y_str = pair.split(',', 1)
            x, y = float(x_str), float(y_str)
            
            # Normalize if needed
            if normalize_coords and x <= 1.0 and y <= 1.0:
                x, y = x * img_width, y * img_height
            
            points.append((max(0, min(x, img_width)), max(0, min(y, img_height))))
        
        # Draw annotation
        if len(points) >= 3:
            draw.polygon(points, outline="blue", width=2)
            draw.text((points[0][0], points[0][1]-15), label, fill="blue")
        
        # Generate normalized coordinates for token
        norm_points = []
        for x, y in points:
            norm_points.append(f"{x/img_width:.3f},{y/img_height:.3f}")
        
        debug_info.append(f"  Polygon: {len(points)} points")
        debug_info.append(f"  Points: {' '.join(norm_points)}")
        
        return f"<|object_ref_start|>{label}<|object_ref_end|> outlined by <|quad_start|>{' '.join(norm_points)}<|quad_end|>"
    
    def _process_object_reference(self, region, debug_info):
        """Process simple object reference"""
        label = region['label']
        debug_info.append(f"  Object reference: {label}")
        return f"<|object_ref_start|>{label}<|object_ref_end|>"
    
    def _tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        np_image = tensor.cpu().numpy()
        if np_image.max() <= 1.0:
            np_image = (np_image * 255).astype(np.uint8)
        else:
            np_image = np_image.astype(np.uint8)
        return Image.fromarray(np_image)
    
    def _pil_to_tensor(self, pil_image):
        """Convert PIL to ComfyUI tensor"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(np_image).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "QwenSpatialTokenGenerator": QwenSpatialTokenGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenSpatialTokenGenerator": "Qwen Spatial Token Generator"
}