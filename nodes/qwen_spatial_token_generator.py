"""
Qwen Spatial Token Generator
Pure spatial token generation without templates or assumptions
"""

import json
import pathlib
from PIL import Image, ImageDraw, ImageColor
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
import logging
import folder_paths
import comfy.utils
import base64
import io

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global storage for JS-Python communication
_node_image_storage = {}

class QwenSpatialTokenGenerator:
    """Generate spatial tokens from image coordinates and labels"""

    def __init__(self):
        self.loaded_image_data = None  # Store loaded image data (base64)
        self.optimized_image_data = None  # Store optimized image from JS
        self.node_id = id(self)  # Unique identifier for this node instance
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image for spatial editing"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Generated spatial prompt (auto-populated from spatial editor)"
                }),
                "output_format": ([
                    "structured_json",
                    "xml_tags",
                    "natural_language", 
                    "traditional_tokens"
                ], {
                    "default": "structured_json",
                    "tooltip": "structured_json: JSON commands (recommended) | xml_tags: HTML-like elements (most native) | natural_language: coordinate sentences | traditional_tokens: legacy format"
                }),
                "debug_mode": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING") 
    RETURN_NAMES = ("annotated_image", "prompt", "debug_info")
    FUNCTION = "generate_tokens"
    CATEGORY = "Qwen/Spatial"
    OUTPUT_NODE = True

    def generate_tokens(self, image, output_format, debug_mode,
                       prompt=""):
        """Generate complete formatted prompt with spatial tokens"""

        logger.info("=== QWEN SPATIAL TOKEN GENERATOR START ===")
        logger.info(f"Inputs - debug_mode: {debug_mode}")
        logger.info(f"Inputs - prompt length: {len(prompt)}")
        
        # Log input image details
        logger.info(f"Input image tensor shape: {image.shape}")
        logger.info(f"Input image tensor dtype: {image.dtype}")
        logger.info(f"Input image tensor min/max: {image.min().item():.4f}/{image.max().item():.4f}")

        debug_info = []
        debug_info.append("=== SPATIAL TOKEN GENERATOR ===")

        try:
            # Convert input image to PIL for processing
            logger.info("Converting input tensor to PIL image")
            debug_info.append("Processing input image")
            pil_image = self._tensor_to_pil(image)
            logger.info(f"PIL image size: {pil_image.size}")
            logger.info(f"PIL image mode: {pil_image.mode}")

            img_width, img_height = pil_image.size
            debug_info.append(f"Working image size: {img_width}x{img_height}")
            logger.info(f"Working with image dimensions: {img_width}x{img_height}")
            
            # Use provided prompt
            if prompt.strip():
                logger.info(f"Using provided prompt: {len(prompt)} characters")
                logger.info("=== PROMPT COORDINATE DEBUGGING ===")
                logger.info(f"Full prompt: {prompt}")
                
                # Check if this is JSON data from JavaScript interface
                try:
                    json_data = json.loads(prompt)
                    logger.info(f"Successfully parsed JSON data: {json_data}")
                    
                    # Check if it's clean JSON commands (already processed by JavaScript)
                    if isinstance(json_data, list) and len(json_data) > 0 and all(isinstance(cmd, dict) for cmd in json_data):
                        logger.info(f"Detected clean JSON commands list from JavaScript (structured_json format)")
                        debug_info.append(f"Using clean JSON commands from JavaScript ({len(json_data)} commands)")
                        
                        # JSON is already clean - just use it as-is and create visualization
                        annotated_image = pil_image.copy()
                        draw = ImageDraw.Draw(annotated_image)
                        logger.info(f"About to draw annotations for {len(json_data)} commands")
                        self._draw_annotations_from_json_commands(json_data, draw, img_width, img_height, debug_info)
                        logger.info("Drew visual annotations from clean JSON commands")
                    
                    # Check if it's a single JSON command
                    elif isinstance(json_data, dict) and any(key in json_data for key in ['bbox', 'point', 'polygon', 'quad', 'target_object', 'action']):
                        logger.info(f"Detected single JSON command from JavaScript (structured_json format)")
                        debug_info.append(f"Using single JSON command from JavaScript")
                        
                        # Wrap single command in a list and process
                        annotated_image = pil_image.copy()
                        draw = ImageDraw.Draw(annotated_image)
                        logger.info(f"About to draw annotations for single command")
                        self._draw_annotations_from_json_commands([json_data], draw, img_width, img_height, debug_info)
                        logger.info("Drew visual annotations from single JSON command")
                    
                    # Check if it's raw region data that needs processing
                    elif isinstance(json_data, dict) and "regions" in json_data:
                        logger.info(f"Detected JavaScript region data in prompt in {output_format} format")
                        debug_info.append(f"Processing {len(json_data['regions'])} regions from JavaScript interface")
                        
                        # Use the regions from JavaScript to generate formatted output
                        spatial_tokens = self._generate_format(json_data['regions'], img_width, img_height, output_format, debug_info)
                        logger.info(f"Generated formatted output: {len(spatial_tokens)} characters")
                        
                        # Replace the raw JSON in prompt with the clean formatted output
                        prompt = spatial_tokens
                        debug_info.append("Replaced raw JSON with clean formatted output in prompt field")
                        
                        # Create annotated image for visualization
                        annotated_image = pil_image.copy()
                        draw = ImageDraw.Draw(annotated_image)
                        self._draw_annotations_from_regions(json_data['regions'], draw, img_width, img_height, debug_info)
                        logger.info("Drew visual annotations from region data")
                    else:
                        raise ValueError("Not recognized JSON format")
                        
                except (json.JSONDecodeError, ValueError, KeyError):
                    # Not JSON region data, handle as regular spatial tokens
                    if output_format == "traditional_tokens":
                        self._debug_spatial_tokens(prompt, img_width, img_height, debug_info)
                        debug_info.append("Using provided traditional spatial tokens - drawing visual annotations")
                        # Parse spatial tokens and draw annotations for visual reference
                        annotated_image = pil_image.copy()
                        draw = ImageDraw.Draw(annotated_image)
                        self._draw_annotations_from_tokens(prompt, draw, img_width, img_height, debug_info)
                        logger.info("Drew visual annotations from spatial tokens")
                    else:
                        debug_info.append(f"Using provided {output_format} spatial tokens")
                        # For non-traditional formats, we can't parse coordinates for visualization yet
                        # Just use the original image
                        annotated_image = pil_image.copy()
                        logger.info(f"Accepted {output_format} format tokens (no coordinate parsing for visualization)")
            else:
                logger.info("No prompt provided")
                debug_info.append("No prompt provided")
                
                # Just use the original image without annotations
                annotated_image = pil_image.copy()
                complete_prompt = ""

            # Create complete prompt
            logger.info("Creating complete prompt...")
            if prompt.strip():
                logger.info("Using provided prompt")
                complete_prompt = prompt.strip()
            else:
                logger.info("No prompt provided")
                complete_prompt = ""
                
            logger.info(f"Complete prompt length: {len(complete_prompt)}")

            debug_info.append(f"Generated prompt: {complete_prompt}")
            debug_info.append(f"Raw output (no template applied)")

            # Convert final image back to tensor
            logger.info("Converting annotated PIL image back to tensor...")
            logger.info(f"Annotated image size: {annotated_image.size}, mode: {annotated_image.mode}")
            final_image = self._pil_to_tensor(annotated_image)
            logger.info(f"Final tensor shape: {final_image.shape}")
            logger.info(f"Final tensor dtype: {final_image.dtype}")
            logger.info(f"Final tensor min/max: {final_image.min().item():.4f}/{final_image.max().item():.4f}")

            debug_text = "\n".join(debug_info) if debug_mode else f"Generated prompt"
            logger.info(f"Debug mode: {debug_mode}, debug text length: {len(debug_text)}")
            
            logger.info(f"Returning outputs:")
            logger.info(f"- annotated_image: {final_image.shape} tensor with visual region markers")  
            logger.info(f"- prompt: {len(complete_prompt)} chars")
            logger.info("=== QWEN SPATIAL TOKEN GENERATOR END ===")

            return (final_image, complete_prompt, debug_text)

        except Exception as e:
            logger.error(f"CRITICAL ERROR in generate_tokens: {str(e)}", exc_info=True)
            error_msg = f"ERROR: {str(e)}"
            debug_info.append(error_msg)
            # Create a blank image as fallback
            logger.info("Creating fallback blank tensor due to error")
            blank_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            logger.info(f"Fallback tensor shape: {blank_tensor.shape}")
            logger.info("=== QWEN SPATIAL TOKEN GENERATOR END (ERROR) ===")
            return (blank_tensor, "", "\n".join(debug_info))

    def _process_bounding_box(self, region, img_width, img_height, normalize_coords, draw, debug_info):
        """Process bounding box coordinates"""
        coords_str = region['coords'].strip() if isinstance(region['coords'], str) else region['coords']
        label = region['label']
        include_object_ref = region.get('includeObjectRef', True)  # Default to True for backward compatibility

        # Handle both string and list formats
        if isinstance(coords_str, str):
            coords = [float(c.strip()) for c in coords_str.split(',')]
        else:
            coords = list(coords_str)

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

        # Draw enhanced annotation for visual reference
        # Use different colors for different regions
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
        color = colors[hash(label) % len(colors)]
        
        # Draw thick, visible rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Draw semi-transparent fill
        from PIL import Image
        overlay = Image.new('RGBA', (int(x2-x1), int(y2-y1)), (*ImageColor.getrgb(color), 30))
        
        if include_object_ref:
            # Draw large, clear label
            from PIL import ImageFont
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Background for text readability
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            draw.rectangle([x1, y1-text_height-4, x1+text_width+8, y1], fill=color)
            draw.text((x1+4, y1-text_height-2), label, fill="white", font=font)
        
        logger.info(f"Drew enhanced annotation: {color} box for '{label}' at ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        # Generate normalized coordinates for token
        norm_x1, norm_y1 = x1 / img_width, y1 / img_height
        norm_x2, norm_y2 = x2 / img_width, y2 / img_height

        debug_info.append(f"  Box coords: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        debug_info.append(f"  Normalized: ({norm_x1:.3f},{norm_y1:.3f},{norm_x2:.3f},{norm_y2:.3f})")
        debug_info.append(f"  Include object_ref: {include_object_ref}")

        if include_object_ref:
            return f"<|object_ref_start|>{label}<|object_ref_end|> at <|box_start|>{norm_x1:.3f},{norm_y1:.3f},{norm_x2:.3f},{norm_y2:.3f}<|box_end|>"
        else:
            return f"<|box_start|>{norm_x1:.3f},{norm_y1:.3f},{norm_x2:.3f},{norm_y2:.3f}<|box_end|>"

    def _process_polygon(self, region, img_width, img_height, normalize_coords, draw, debug_info):
        """Process polygon coordinates"""
        coords_data = region['coords']
        label = region['label']
        include_object_ref = region.get('includeObjectRef', True)  # Default to True for backward compatibility

        # Handle both string and list formats
        if isinstance(coords_data, str):
            # Parse space-separated x,y pairs from string
            coord_pairs = coords_data.strip().split()
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
        else:
            # Handle list of [x,y] pairs from JavaScript
            if len(coords_data) < 3:
                raise ValueError(f"Polygon needs at least 3 points, got {len(coords_data)}")

            points = []
            for coord in coords_data:
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    x, y = coord[0], coord[1]
                else:
                    raise ValueError(f"Invalid coordinate format: {coord}")

                # Normalize if needed
                if normalize_coords and x <= 1.0 and y <= 1.0:
                    x, y = x * img_width, y * img_height

                points.append((max(0, min(x, img_width)), max(0, min(y, img_height))))

        # Draw annotation
        if len(points) >= 3:
            draw.polygon(points, outline="blue", width=2)
            if include_object_ref:
                draw.text((points[0][0], points[0][1]-15), label, fill="blue")

        # Generate normalized coordinates for token
        norm_points = []
        for x, y in points:
            norm_points.append(f"{x/img_width:.3f},{y/img_height:.3f}")

        debug_info.append(f"  Polygon: {len(points)} points")
        debug_info.append(f"  Points: {' '.join(norm_points)}")
        debug_info.append(f"  Include object_ref: {include_object_ref}")

        if include_object_ref:
            return f"<|object_ref_start|>{label}<|object_ref_end|> outlined by <|quad_start|>{' '.join(norm_points)}<|quad_end|>"
        else:
            return f"<|quad_start|>{' '.join(norm_points)}<|quad_end|>"

    def _process_object_reference(self, region, debug_info):
        """Process simple object reference"""
        label = region['label']
        debug_info.append(f"  Object reference: {label}")
        return f"<|object_ref_start|>{label}<|object_ref_end|>"

    def _tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL"""
        logger.debug(f"_tensor_to_pil: Input tensor shape: {tensor.shape}")
        if len(tensor.shape) == 4:
            logger.debug("_tensor_to_pil: Removing batch dimension")
            tensor = tensor[0]
        logger.debug(f"_tensor_to_pil: Working tensor shape: {tensor.shape}")
        np_image = tensor.cpu().numpy()
        logger.debug(f"_tensor_to_pil: Numpy array shape: {np_image.shape}, dtype: {np_image.dtype}")
        logger.debug(f"_tensor_to_pil: Numpy array min/max: {np_image.min():.4f}/{np_image.max():.4f}")
        if np_image.max() <= 1.0:
            logger.debug("_tensor_to_pil: Scaling from 0-1 to 0-255")
            np_image = (np_image * 255).astype(np.uint8)
        else:
            logger.debug("_tensor_to_pil: Converting to uint8 without scaling")
            np_image = np_image.astype(np.uint8)
        logger.debug(f"_tensor_to_pil: Final numpy array dtype: {np_image.dtype}")
        pil_image = Image.fromarray(np_image)
        logger.debug(f"_tensor_to_pil: Output PIL image size: {pil_image.size}, mode: {pil_image.mode}")
        return pil_image

    def _pil_to_tensor(self, pil_image):
        """Convert PIL to ComfyUI tensor"""
        logger.debug(f"_pil_to_tensor: Input PIL image size: {pil_image.size}, mode: {pil_image.mode}")
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        logger.debug(f"_pil_to_tensor: Numpy array shape: {np_image.shape}, dtype: {np_image.dtype}")
        logger.debug(f"_pil_to_tensor: Numpy array min/max: {np_image.min():.4f}/{np_image.max():.4f}")
        tensor = torch.from_numpy(np_image).unsqueeze(0)
        logger.debug(f"_pil_to_tensor: Output tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        return tensor

    def _base64_to_pil(self, base64_data):
        """Convert base64 data to PIL image"""
        logger.debug(f"_base64_to_pil: Input data length: {len(base64_data)} characters")
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',', 1)[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_data)
            logger.debug(f"_base64_to_pil: Decoded to {len(image_bytes)} bytes")
            
            # Create PIL image from bytes
            pil_image = Image.open(io.BytesIO(image_bytes))
            logger.debug(f"_base64_to_pil: Created PIL image size: {pil_image.size}, mode: {pil_image.mode}")
            return pil_image
        except Exception as e:
            logger.error(f"_base64_to_pil: Error converting base64 to PIL: {e}")
            raise

    def _pil_to_base64(self, pil_image):
        """Convert PIL image to base64 data URL"""
        logger.debug(f"_pil_to_base64: Input PIL image size: {pil_image.size}, mode: {pil_image.mode}")
        try:
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data_url = f"data:image/png;base64,{base64_data}"
            logger.debug(f"_pil_to_base64: Created data URL length: {len(data_url)} characters")
            return data_url
        except Exception as e:
            logger.error(f"_pil_to_base64: Error converting PIL to base64: {e}")
            raise

    def load_image_from_file(self, file_path):
        """Load image from file path and store as base64"""
        logger.info(f"Loading image from file: {file_path}")
        try:
            pil_image = Image.open(file_path)
            logger.info(f"Loaded image size: {pil_image.size}, mode: {pil_image.mode}")
            self.loaded_image_data = self._pil_to_base64(pil_image)
            logger.info("Image stored as base64 data")
            return True
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return False

    def set_optimized_image(self, base64_data):
        """Set optimized image data from JS interface"""
        logger.info(f"Setting optimized image data: {len(base64_data)} characters")
        self.optimized_image_data = base64_data
        logger.info("Optimized image data stored")

    def _debug_spatial_tokens(self, spatial_tokens, img_width, img_height, debug_info):
        """Parse and debug spatial tokens to verify coordinate accuracy"""
        import re
        
        logger.info("Parsing spatial tokens for coordinate debugging...")
        
        # Calculate native ViT dimensions (what the model actually sees)
        native_width = ((img_width + 27) // 28) * 28  # Round up to multiple of 28
        native_height = ((img_height + 27) // 28) * 28
        
        logger.info(f"Image dimensions: {img_width}x{img_height}")
        logger.info(f"Native ViT dimensions (multiple of 28): {native_width}x{native_height}")
        logger.info(f"Scale factors: X={native_width/img_width:.3f}, Y={native_height/img_height:.3f}")
        
        # Extract bounding box tokens with absolute pixel coordinates
        box_pattern = r'<\|box_start\|>(\d+),(\d+),(\d+),(\d+)<\|box_end\|>'
        box_matches = re.findall(box_pattern, spatial_tokens)
        
        # Extract object reference + box tokens with absolute pixels
        obj_box_pattern = r'<\|object_ref_start\|>([^<]+)<\|object_ref_end\|>\s*at\s*<\|box_start\|>(\d+),(\d+),(\d+),(\d+)<\|box_end\|>'
        obj_box_matches = re.findall(obj_box_pattern, spatial_tokens)
        
        # Extract quad tokens with native coordinates
        quad_pattern = r'<\|quad_start\|>(\([^)]+\)(?:,\([^)]+\))*)<\|quad_end\|>'
        quad_matches = re.findall(quad_pattern, spatial_tokens)
        
        logger.info(f"Found {len(box_matches)} standalone boxes, {len(obj_box_matches)} object+box combinations, {len(quad_matches)} quads")
        
        for i, (x1_str, y1_str, x2_str, y2_str) in enumerate(box_matches):
            x1, y1, x2, y2 = int(x1_str), int(y1_str), int(x2_str), int(y2_str)
            
            # Verify coordinates are within native ViT bounds
            valid_coords = (0 <= x1 <= native_width and 0 <= y1 <= native_height and 
                          0 <= x2 <= native_width and 0 <= y2 <= native_height)
            
            # Calculate coverage on native dimensions
            coverage_w = ((x2-x1)/native_width)*100
            coverage_h = ((y2-y1)/native_height)*100
            
            logger.info(f"=== BOX {i+1} COORDINATE VERIFICATION (NATIVE ViT PIXELS) ===")
            logger.info(f"Native pixel coords: ({x1},{y1},{x2},{y2})")
            logger.info(f"Box size: {x2-x1}x{y2-y1} native pixels")
            logger.info(f"Valid coordinates: {valid_coords}")
            logger.info(f"Coverage: {coverage_w:.1f}% width, {coverage_h:.1f}% height")
            
            debug_info.append(f"BOX {i+1}: native_pixels({x1},{y1},{x2},{y2}) valid={valid_coords}")
        
        for i, (label, x1_str, y1_str, x2_str, y2_str) in enumerate(obj_box_matches):
            x1, y1, x2, y2 = int(x1_str), int(y1_str), int(x2_str), int(y2_str)
            
            # Verify coordinates are within native ViT bounds
            valid_coords = (0 <= x1 <= native_width and 0 <= y1 <= native_height and 
                          0 <= x2 <= native_width and 0 <= y2 <= native_height)
            
            # Calculate coverage on native dimensions
            coverage_w = ((x2-x1)/native_width)*100
            coverage_h = ((y2-y1)/native_height)*100
            
            logger.info(f"=== OBJECT+BOX {i+1} '{label.strip()}' COORDINATE VERIFICATION (NATIVE ViT PIXELS) ===")
            logger.info(f"Native pixel coords: ({x1},{y1},{x2},{y2})")
            logger.info(f"Box size: {x2-x1}x{y2-y1} native pixels")
            logger.info(f"Valid coordinates: {valid_coords}")
            logger.info(f"Coverage: {coverage_w:.1f}% width, {coverage_h:.1f}% height")
            
            debug_info.append(f"OBJECT '{label.strip()}': native_pixels({x1},{y1},{x2},{y2}) valid={valid_coords}")
        
        # Debug quad tokens
        for i, quad_coords in enumerate(quad_matches):
            logger.info(f"=== QUAD {i+1} COORDINATE VERIFICATION ===")
            logger.info(f"Quad coordinates: {quad_coords}")
            debug_info.append(f"QUAD {i+1}: {quad_coords}")
        
        logger.info("=== SPATIAL TOKENS COORDINATE DEBUGGING COMPLETE ===")

    def _draw_annotations_from_regions(self, regions, draw, img_width, img_height, debug_info):
        """Draw visual annotations from JavaScript region data"""
        
        logger.info("Drawing annotations from JavaScript region data...")
        
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
        
        for i, region in enumerate(regions):
            color = colors[i % len(colors)]
            
            if region['type'] == 'bounding_box':
                x1, y1, x2, y2 = region['coords']
                
                # Draw enhanced annotation
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                
                # Draw label if available
                if region.get('label'):
                    from PIL import ImageFont
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                    
                    # Background for text readability
                    text_bbox = draw.textbbox((0, 0), region['label'], font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    draw.rectangle([x1, y1-text_height-4, x1+text_width+8, y1], fill=color)
                    draw.text((x1+4, y1-text_height-2), region['label'], fill="white", font=font)
                
                logger.info(f"Drew bounding box: {color} for '{region.get('label', 'unlabeled')}' at ({x1},{y1},{x2},{y2})")
                
            elif region['type'] == 'polygon':
                if len(region['coords']) >= 3:
                    # Draw polygon
                    points = [(x, y) for x, y in region['coords']]
                    draw.polygon(points, outline=color, width=2)
                    
                    # Draw label at first point
                    if region.get('label') and points:
                        draw.text((points[0][0], points[0][1]-15), region['label'], fill=color)
                    
                    logger.info(f"Drew polygon: {color} for '{region.get('label', 'unlabeled')}' with {len(points)} points")
                
            elif region['type'] == 'object_reference':
                if len(region['coords']) >= 2:
                    x, y = region['coords'][:2]  # Take first two coordinates
                    
                    # Draw a colored circle
                    draw.ellipse([x-8, y-8, x+8, y+8], outline=color, fill=color, width=4)
                    
                    # Add a white center dot
                    draw.ellipse([x-3, y-3, x+3, y+3], fill="white")
                    
                    # Label
                    if region.get('label'):
                        draw.text((x + 12, y - 5), region['label'], fill=color)
                    
                    logger.info(f"Drew object reference: {color} for '{region.get('label', 'unlabeled')}' at ({x},{y})")
        
        debug_info.append(f"Drew {len(regions)} visual annotations from region data")

    def _draw_annotations_from_json_commands(self, json_commands, draw, img_width, img_height, debug_info):
        """Draw visual annotations from clean JSON commands"""
        
        logger.info("Drawing annotations from clean JSON commands...")
        
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
        
        annotations_drawn = 0
        
        for i, command in enumerate(json_commands):
            color = colors[i % len(colors)]
            logger.info(f"Processing command {i}: {command}")
            
            if 'bbox' in command:
                x1, y1, x2, y2 = command['bbox']
                
                # Draw enhanced annotation
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                annotations_drawn += 1
                
                # Draw label if available
                label = command.get('target_object') or command.get('target') or f"Region {i+1}"
                from PIL import ImageFont
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                # Background for text readability
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                draw.rectangle([x1, y1-text_height-4, x1+text_width+8, y1], fill=color)
                draw.text((x1+4, y1-text_height-2), label, fill="white", font=font)
                
                logger.info(f"Drew bbox command: {color} for '{label}' at ({x1},{y1},{x2},{y2})")
                
            elif 'polygon' in command:
                if len(command['polygon']) >= 3:
                    # Draw polygon
                    points = [(x, y) for x, y in command['polygon']]
                    draw.polygon(points, outline=color, width=2)
                    annotations_drawn += 1
                    
                    # Draw label at first point
                    label = command.get('target_object') or command.get('target') or f"Polygon {i+1}"
                    if points:
                        draw.text((points[0][0], points[0][1]-15), label, fill=color)
                    
                    logger.info(f"Drew polygon command: {color} for '{label}' with {len(points)} points")
                
            elif 'point' in command:
                if len(command['point']) >= 2:
                    x, y = command['point'][:2]
                    
                    # Draw a colored circle (outline first, then fill)
                    draw.ellipse([x-10, y-10, x+10, y+10], outline=color, width=4)
                    draw.ellipse([x-8, y-8, x+8, y+8], fill=color)
                    annotations_drawn += 1
                    
                    # Add a white center dot
                    draw.ellipse([x-3, y-3, x+3, y+3], fill="white")
                    
                    # Label with font
                    label = command.get('target_object') or command.get('target') or f"Point {i+1}"
                    from PIL import ImageFont
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                    
                    draw.text((x + 15, y - 8), label, fill=color, font=font)
                    
                    logger.info(f"Drew reference point command: {color} for '{label}' at ({x},{y})")
                    
            elif 'quad' in command:
                if len(command['quad']) >= 8:
                    # Convert flat list to coordinate pairs
                    points = [(command['quad'][i], command['quad'][i+1]) for i in range(0, len(command['quad']), 2)]
                    if len(points) >= 3:
                        # Draw polygon
                        draw.polygon(points, outline=color, width=2)
                        annotations_drawn += 1
                        
                        # Draw label at first point
                        label = command.get('target_object') or command.get('target') or f"Quad {i+1}"
                        if points:
                            draw.text((points[0][0], points[0][1]-15), label, fill=color)
                        
                        logger.info(f"Drew quad command: {color} for '{label}' with {len(points)} points")
            else:
                logger.info(f"Command {i} has no recognized coordinate fields (bbox, polygon, point, quad)")
        
        logger.info(f"Total annotations drawn: {annotations_drawn} out of {len(json_commands)} commands")
        debug_info.append(f"Drew {annotations_drawn} visual annotations from {len(json_commands)} JSON commands")

    def _draw_annotations_from_tokens(self, spatial_tokens, draw, img_width, img_height, debug_info):
        """Parse spatial tokens and draw visual annotations"""
        import re
        
        logger.info("Drawing annotations from spatial tokens...")
        
        # Calculate native ViT dimensions for coordinate scaling
        native_width = ((img_width + 27) // 28) * 28  
        native_height = ((img_height + 27) // 28) * 28
        
        scale_x = img_width / native_width  # Scale back from native to display
        scale_y = img_height / native_height
        
        # Extract bounding box tokens
        box_pattern = r'<\|box_start\|>(\d+),(\d+),(\d+),(\d+)<\|box_end\|>'
        box_matches = re.findall(box_pattern, spatial_tokens)
        
        # Extract object reference + box tokens  
        obj_box_pattern = r'<\|object_ref_start\|>([^<]+)<\|object_ref_end\|>\s*at\s*<\|box_start\|>(\d+),(\d+),(\d+),(\d+)<\|box_end\|>'
        obj_box_matches = re.findall(obj_box_pattern, spatial_tokens)
        
        # Extract quad tokens
        quad_pattern = r'<\|quad_start\|>(\([^)]+\)(?:,\([^)]+\))*)<\|quad_end\|>'
        quad_matches = re.findall(quad_pattern, spatial_tokens)
        
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
        
        # Draw standalone boxes
        for i, (x1_str, y1_str, x2_str, y2_str) in enumerate(box_matches):
            x1, y1, x2, y2 = int(x1_str), int(y1_str), int(x2_str), int(y2_str)
            
            # Scale from native to display coordinates
            display_x1 = x1 * scale_x
            display_y1 = y1 * scale_y
            display_x2 = x2 * scale_x
            display_y2 = y2 * scale_y
            
            color = colors[i % len(colors)]
            draw.rectangle([display_x1, display_y1, display_x2, display_y2], outline=color, width=4)
            logger.info(f"Drew standalone box {i+1}: {color} at ({display_x1:.0f},{display_y1:.0f},{display_x2:.0f},{display_y2:.0f})")
        
        # Draw object+box combinations
        for i, (label, x1_str, y1_str, x2_str, y2_str) in enumerate(obj_box_matches):
            x1, y1, x2, y2 = int(x1_str), int(y1_str), int(x2_str), int(y2_str)
            
            # Scale from native to display coordinates
            display_x1 = x1 * scale_x
            display_y1 = y1 * scale_y
            display_x2 = x2 * scale_x
            display_y2 = y2 * scale_y
            
            color = colors[(i + len(box_matches)) % len(colors)]
            draw.rectangle([display_x1, display_y1, display_x2, display_y2], outline=color, width=4)
            
            # Draw label with background
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((0, 0), label.strip(), font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            draw.rectangle([display_x1, display_y1-text_height-4, display_x1+text_width+8, display_y1], fill=color)
            draw.text((display_x1+4, display_y1-text_height-2), label.strip(), fill="white", font=font)
            
            logger.info(f"Drew object+box {i+1}: {color} '{label.strip()}' at ({display_x1:.0f},{display_y1:.0f},{display_x2:.0f},{display_y2:.0f})")
        
        # Extract standalone object references
        obj_ref_pattern = r'<\|object_ref_start\|>([^<]+)<\|object_ref_end\|>'
        obj_ref_matches = re.findall(obj_ref_pattern, spatial_tokens)
        
        # Draw standalone object references as points (no coordinates, just labels)
        for i, label in enumerate(obj_ref_matches):
            # Skip if this label also has a box (already drawn above)
            if not any(label.strip() in match[0] for match in obj_box_matches):
                color = colors[(i + len(box_matches) + len(obj_box_matches)) % len(colors)]
                
                # Draw a prominent circle in center of image as placeholder
                center_x, center_y = img_width // 2, img_height // 2
                radius = 20
                
                # Draw circle
                draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], 
                           outline=color, fill=color, width=4)
                
                # Draw label
                try:
                    from PIL import ImageFont
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                draw.text((center_x+radius+5, center_y-10), label.strip(), fill=color, font=font)
                logger.info(f"Drew object reference {i+1}: {color} '{label.strip()}' at center ({center_x},{center_y})")
        
        # TODO: Add quad drawing support
        if quad_matches:
            logger.info(f"Found {len(quad_matches)} quad tokens - drawing not yet implemented")
            
        total_annotations = len(box_matches) + len(obj_box_matches) + len([l for l in obj_ref_matches if not any(l.strip() in m[0] for m in obj_box_matches)])
        debug_info.append(f"Drew {total_annotations} visual annotations")
        logger.info(f"Total annotations drawn: {total_annotations}")

    def _get_js_optimized_image(self):
        """Get optimized image data from JavaScript bridge via global storage"""
        try:
            logger.debug("Checking for JS bridge optimized image data...")
            
            # Check global storage for this node's image data
            node_key = str(self.node_id)
            if node_key in _node_image_storage:
                logger.info(f"Found optimized image data for node {node_key} in global storage")
                return _node_image_storage[node_key]
            
            # Try to get from JavaScript execution context if available
            try:
                import js2py
                if hasattr(js2py, 'eval_js'):
                    js_storage = js2py.eval_js(f'window._qwen_spatial_storage && window._qwen_spatial_storage["{node_key}"]')
                    if js_storage:
                        logger.info(f"Found optimized image data for node {node_key} in JS context")
                        # Store it for future use
                        _node_image_storage[node_key] = js_storage
                        return js_storage
            except:
                pass  # js2py not available or JS context not accessible
            
            logger.debug(f"No optimized image found for node {node_key}")
            return None
        except Exception as e:
            logger.debug(f"Error accessing JS bridge data: {e}")
            return None

    @classmethod 
    def store_optimized_image(cls, node_id, base64_data):
        """Store optimized image data from JavaScript (called via bridge)"""
        logger.info(f"Storing optimized image for node {node_id}: {len(base64_data)} characters")
        _node_image_storage[str(node_id)] = base64_data

    def _generate_format(self, regions, img_width, img_height, output_format, debug_info):
        """Generate spatial tokens in the specified format"""
        
        if output_format == "structured_json":
            return self._generate_structured_json(regions, img_width, img_height, debug_info)
        elif output_format == "xml_tags":
            return self._generate_xml_tags(regions, img_width, img_height, debug_info)
        elif output_format == "natural_language":
            return self._generate_natural_language(regions, img_width, img_height, debug_info)
        else:
            debug_info.append(f"Unknown format: {output_format}, falling back to traditional tokens")
            return ""

    def _generate_structured_json(self, regions, img_width, img_height, debug_info):
        """Generate structured JSON commands"""
        
        # Calculate native ViT dimensions
        native_width = ((img_width + 27) // 28) * 28
        native_height = ((img_height + 27) // 28) * 28
        
        commands = []
        
        for region in regions:
            if region['type'] == 'bounding_box':
                x1, y1, x2, y2 = region['coords']
                
                # Scale to native coordinates
                scale_x = native_width / img_width
                scale_y = native_height / img_height
                native_x1 = round(x1 * scale_x)
                native_y1 = round(y1 * scale_y)
                native_x2 = round(x2 * scale_x)
                native_y2 = round(y2 * scale_y)
                
                command = {
                    "action": "edit_region",
                    "target": region['label'],
                    "bbox": [native_x1, native_y1, native_x2, native_y2],
                    "instruction": f"modify the {region['label']}",
                    "preserve": "background, lighting, other objects"
                }
                
                if not region.get('includeObjectRef', True):
                    # Remove target if object reference disabled
                    del command['target']
                    command['action'] = "edit_area"
                
                commands.append(command)
                
            elif region['type'] == 'polygon':
                # Convert polygon points to native coordinates
                native_points = []
                scale_x = native_width / img_width
                scale_y = native_height / img_height
                
                for x, y in region['coords']:
                    native_x = round(x * scale_x)
                    native_y = round(y * scale_y)
                    native_points.append([native_x, native_y])
                
                command = {
                    "action": "edit_polygon",
                    "target": region['label'],
                    "polygon": native_points,
                    "instruction": f"modify the {region['label']}",
                    "preserve": "background, lighting, other objects"
                }
                
                if not region.get('includeObjectRef', True):
                    del command['target']
                    command['action'] = "edit_shape"
                
                commands.append(command)
                
            elif region['type'] == 'object_reference':
                command = {
                    "action": "reference_object", 
                    "target": region['label'],
                    "instruction": f"focus on the {region['label']}"
                }
                commands.append(command)
        
        if len(commands) == 1:
            result = json.dumps(commands[0], indent=2)
        else:
            result = json.dumps({"directives": commands}, indent=2)
        
        debug_info.append(f"Generated {len(commands)} JSON command(s)")
        return result

    def _generate_xml_tags(self, regions, img_width, img_height, debug_info):
        """Generate XML-like tags (most native to Qwen training)"""
        
        # Calculate native ViT dimensions
        native_width = ((img_width + 27) // 28) * 28
        native_height = ((img_height + 27) // 28) * 28
        
        xml_elements = []
        
        for region in regions:
            if region['type'] == 'bounding_box':
                x1, y1, x2, y2 = region['coords']
                
                # Scale to native coordinates
                scale_x = native_width / img_width
                scale_y = native_height / img_height
                native_x1 = round(x1 * scale_x)
                native_y1 = round(y1 * scale_y)
                native_x2 = round(x2 * scale_x)
                native_y2 = round(y2 * scale_y)
                
                if region.get('includeObjectRef', True):
                    xml_element = f'<region data-bbox="{native_x1},{native_y1},{native_x2},{native_y2}">\n'
                    xml_element += f'  <target>{region["label"]}</target>\n'
                    xml_element += f'  <action>edit_region</action>\n'
                    xml_element += f'  <instruction>modify the {region["label"]}</instruction>\n'
                    xml_element += f'  <preserve>background, lighting, other objects</preserve>\n'
                    xml_element += '</region>'
                else:
                    xml_element = f'<region data-bbox="{native_x1},{native_y1},{native_x2},{native_y2}">\n'
                    xml_element += f'  <action>edit_area</action>\n'
                    xml_element += f'  <instruction>modify this area</instruction>\n'
                    xml_element += f'  <preserve>background, lighting, other objects</preserve>\n'
                    xml_element += '</region>'
                
                xml_elements.append(xml_element)
                
            elif region['type'] == 'polygon':
                # Format polygon points as coordinate pairs
                scale_x = native_width / img_width
                scale_y = native_height / img_height
                coord_pairs = []
                
                for x, y in region['coords']:
                    native_x = round(x * scale_x)
                    native_y = round(y * scale_y)
                    coord_pairs.append(f"({native_x},{native_y})")
                
                coords_str = ",".join(coord_pairs)
                
                if region.get('includeObjectRef', True):
                    xml_element = f'<region data-polygon="{coords_str}">\n'
                    xml_element += f'  <target>{region["label"]}</target>\n'
                    xml_element += f'  <action>edit_polygon</action>\n'
                    xml_element += f'  <instruction>modify the {region["label"]}</instruction>\n'
                    xml_element += f'  <preserve>background, lighting, other objects</preserve>\n'
                    xml_element += '</region>'
                else:
                    xml_element = f'<region data-polygon="{coords_str}">\n'
                    xml_element += f'  <action>edit_shape</action>\n'
                    xml_element += f'  <instruction>modify this shape</instruction>\n'
                    xml_element += f'  <preserve>background, lighting, other objects</preserve>\n'
                    xml_element += '</region>'
                
                xml_elements.append(xml_element)
                
            elif region['type'] == 'object_reference':
                xml_element = f'<reference>\n'
                xml_element += f'  <target>{region["label"]}</target>\n'
                xml_element += f'  <action>reference_object</action>\n'
                xml_element += f'  <instruction>focus on the {region["label"]}</instruction>\n'
                xml_element += '</reference>'
                xml_elements.append(xml_element)
        
        result = "\n\n".join(xml_elements)
        debug_info.append(f"Generated {len(xml_elements)} XML element(s)")
        return result

    def _generate_natural_language(self, regions, img_width, img_height, debug_info):
        """Generate natural language with coordinate references"""
        
        # Calculate native ViT dimensions
        native_width = ((img_width + 27) // 28) * 28
        native_height = ((img_height + 27) // 28) * 28
        
        sentences = []
        
        for region in regions:
            if region['type'] == 'bounding_box':
                x1, y1, x2, y2 = region['coords']
                
                # Scale to native coordinates
                scale_x = native_width / img_width
                scale_y = native_height / img_height
                native_x1 = round(x1 * scale_x)
                native_y1 = round(y1 * scale_y)
                native_x2 = round(x2 * scale_x)
                native_y2 = round(y2 * scale_y)
                
                bbox_str = f"[{native_x1},{native_y1},{native_x2},{native_y2}]"
                
                if region.get('includeObjectRef', True):
                    sentence = f"Within the bounding box {bbox_str}, modify the {region['label']}. Preserve the background, lighting, and other objects."
                else:
                    sentence = f"Within the bounding box {bbox_str}, make changes to this area. Preserve the background, lighting, and other objects."
                
                sentences.append(sentence)
                
            elif region['type'] == 'polygon':
                # Format polygon as coordinate list
                scale_x = native_width / img_width
                scale_y = native_height / img_height
                coord_pairs = []
                
                for x, y in region['coords']:
                    native_x = round(x * scale_x)
                    native_y = round(y * scale_y)
                    coord_pairs.append(f"({native_x},{native_y})")
                
                coords_str = ",".join(coord_pairs)
                
                if region.get('includeObjectRef', True):
                    sentence = f"Within the polygon defined by points {coords_str}, modify the {region['label']}. Preserve the background, lighting, and other objects."
                else:
                    sentence = f"Within the polygon defined by points {coords_str}, make changes to this shape. Preserve the background, lighting, and other objects."
                
                sentences.append(sentence)
                
            elif region['type'] == 'object_reference':
                sentence = f"Focus on the {region['label']} in the image."
                sentences.append(sentence)
        
        result = " ".join(sentences)
        debug_info.append(f"Generated {len(sentences)} natural language instruction(s)")
        return result

    def _apply_template(self, prompt, template_mode, debug_info):
        """Apply template formatting to the complete prompt"""

        debug_info.append(f"Applying template: {template_mode}")

        if template_mode == "raw":
            debug_info.append("Using raw prompt (no template)")
            return prompt

        # Template definitions (simplified from QwenTemplateBuilderV2)
        templates = {
            "default_edit": {
                "system": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
                "vision": True
            }
        }

        if template_mode in templates:
            template = templates[template_mode]
            system_prompt = template["system"]
            include_vision = template["vision"]

            # Build chat template
            result = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            result += f"<|im_start|>user\n"

            if include_vision:
                result += "<|vision_start|><|image_pad|><|vision_end|>"

            result += f"{prompt}<|im_end|>\n"
            result += f"<|im_start|>assistant\n"

            debug_info.append(f"Applied {template_mode} template with vision: {include_vision}")
            return result
        else:
            debug_info.append(f"Unknown template mode: {template_mode}, using raw")
            return prompt



NODE_CLASS_MAPPINGS = {
    "QwenSpatialTokenGenerator": QwenSpatialTokenGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenSpatialTokenGenerator": "Qwen Spatial Token Generator"
}

# Expose class methods to JavaScript bridge
def store_optimized_image_bridge(node_id, base64_data):
    """JavaScript bridge function to store optimized image data"""
    logger.info(f"JS Bridge: Storing optimized image for node {node_id}: {len(base64_data)} characters")
    QwenSpatialTokenGenerator.store_optimized_image(node_id, base64_data)
    return {"status": "success"}
