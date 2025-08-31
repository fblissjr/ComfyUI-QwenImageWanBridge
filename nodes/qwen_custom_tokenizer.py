"""
Custom Tokenizer for Qwen2.5-VL Multi-Frame Support
Bypasses ComfyUI's single-image limitation
"""

import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import comfy.model_management
from transformers import Qwen2Tokenizer

logger = logging.getLogger(__name__)


class QwenMultiFrameTokenizer:
    """
    Custom tokenizer that properly handles multiple frames for Qwen2.5-VL.
    This replaces ComfyUI's tokenizer to enable true multi-frame vision.
    """
    
    def __init__(self):
        # Initialize the Qwen tokenizer
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 
            "../../../models/text_encoders/qwen25_tokenizer"
        )
        
        # Try multiple paths
        if not os.path.exists(tokenizer_path):
            # Try ComfyUI's text_encoders path
            tokenizer_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../../comfy/text_encoders/qwen25_tokenizer"
            )
        
        if not os.path.exists(tokenizer_path):
            # Try relative to ComfyUI root
            try:
                import comfy
                if hasattr(comfy, '__file__'):
                    comfy_root = os.path.dirname(os.path.dirname(comfy.__file__))
                    tokenizer_path = os.path.join(
                        comfy_root,
                        "ComfyUI/comfy/text_encoders/qwen25_tokenizer"
                    )
            except:
                pass
        
        try:
            self.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Loaded Qwen tokenizer from {tokenizer_path}")
        except:
            # Fallback to basic tokenizer
            logger.warning("Could not load Qwen tokenizer, using fallback")
            self.tokenizer = None
        
        # Special tokens for Qwen2.5-VL
        self.vision_start_token = 151655  # <|vision_start|>
        self.image_pad_token = 151859     # <|image_pad|>
        self.vision_end_token = 151656    # <|vision_end|>
        self.im_start_token = 151644      # <|im_start|>
        self.im_end_token = 151645        # <|im_end|>
        
        # Spatial reference tokens
        self.object_ref_start = 151646
        self.object_ref_end = 151647
        self.box_start = 151648
        self.box_end = 151649
        
    def create_multiframe_tokens(self, text: str, frames: List[torch.Tensor], 
                                 vision_patches: torch.Tensor,
                                 grid_thw: torch.Tensor,
                                 debug: bool = False) -> Dict[str, Any]:
        """
        Create tokens with proper multi-frame vision embedding.
        
        Args:
            text: The text prompt
            frames: List of frame tensors (for metadata)
            vision_patches: Processed vision patches [seq_len, patch_dim]
            grid_thw: Grid dimensions [T, H, W]
            debug: Enable debug logging
            
        Returns:
            Token dictionary compatible with ComfyUI's encode_from_tokens
        """
        
        # The text should already be formatted with proper template
        # We just need to ensure vision tokens are in the right place
        full_prompt = text  # Use text as-is since it's already formatted
        
        # Tokenize the text
        if self.tokenizer:
            text_tokens = self.tokenizer.encode(full_prompt)
        else:
            # Fallback tokenization (basic)
            text_tokens = self._basic_tokenize(full_prompt)
        
        if debug:
            logger.info(f"[Tokenizer] Text tokens: {len(text_tokens)}")
            logger.info(f"[Tokenizer] Vision patches shape: {vision_patches.shape}")
            logger.info(f"[Tokenizer] Grid THW: {grid_thw}")
        
        # Find the image_pad token position
        image_pad_idx = None
        for i, token in enumerate(text_tokens):
            if token == self.image_pad_token:
                image_pad_idx = i
                break
        
        if image_pad_idx is None:
            logger.warning("[Tokenizer] No image_pad token found, appending vision at end")
            image_pad_idx = len(text_tokens) - 1
        
        # Create the token structure with vision embedding
        # This is the key - we inject our multi-frame vision patches
        token_weights = []
        
        for i, token_id in enumerate(text_tokens):
            if i == image_pad_idx:
                # Replace image_pad with our vision patches
                # Mark it as a special multi-frame vision token
                vision_embed = {
                    "type": "multiframe_vision",  # Critical: custom type for multiframe
                    "data": vision_patches,        # Pre-processed patches [seq_len, patch_dim]
                    "grid_thw": grid_thw,          # Grid dimensions [T, H, W] where T=num_frames
                    "num_frames": len(frames),     # Number of temporal frames
                    # Do NOT set original_type to "image" - that triggers wrong pipeline
                }
                token_weights.append((vision_embed, 1.0, 0))
            else:
                # Regular text token
                token_weights.append((token_id, 1.0, 0))
        
        if debug:
            logger.info(f"[Tokenizer] Created {len(token_weights)} tokens with multi-frame vision at position {image_pad_idx}")
        
        # Return in ComfyUI's expected format
        return {
            "qwen25_7b": [token_weights],
            "multiframe_metadata": {
                "vision_patches": vision_patches,
                "grid_thw": grid_thw,
                "num_frames": len(frames),
                "image_pad_position": image_pad_idx
            }
        }
    
    def _basic_tokenize(self, text: str) -> List[int]:
        """
        Basic fallback tokenization when Qwen tokenizer isn't available.
        """
        # This is a very basic implementation
        # In practice, we'd need the actual tokenizer
        tokens = []
        
        # Add special tokens based on the text
        if "<|im_start|>" in text:
            tokens.append(self.im_start_token)
        if "<|vision_start|>" in text:
            tokens.append(self.vision_start_token)
        if "<|image_pad|>" in text:
            tokens.append(self.image_pad_token)
        if "<|vision_end|>" in text:
            tokens.append(self.vision_end_token)
        
        # Add some text tokens (simplified)
        words = text.replace("<|", " <|").replace("|>", "|> ").split()
        for word in words:
            if not word.startswith("<|"):
                # Assign arbitrary token IDs for words (this is oversimplified)
                tokens.append(hash(word) % 50000 + 1000)
        
        if "<|im_end|>" in text:
            tokens.append(self.im_end_token)
        
        return tokens


class MultiFrameVisionEmbedder:
    """
    Handles the embedding of multi-frame vision patches into the model.
    This is what actually makes the frames visible to the model.
    """
    
    def __init__(self, hidden_size: int = 3584):
        self.hidden_size = hidden_size
        
    def embed_vision_patches(self, vision_patches: torch.Tensor, 
                            grid_thw: torch.Tensor,
                            vision_model: Optional[Any] = None) -> torch.Tensor:
        """
        Convert vision patches to embeddings.
        
        Args:
            vision_patches: Flattened patches [seq_len, patch_dim]
            grid_thw: Grid dimensions [T, H, W]
            vision_model: Optional vision encoder model
            
        Returns:
            Vision embeddings [seq_len, hidden_size]
        """
        device = vision_patches.device
        seq_len = vision_patches.shape[0]
        
        if vision_model is not None:
            try:
                # Use the actual vision model if available
                # Add batch dimension
                batched_patches = vision_patches.unsqueeze(0)
                batched_grid = grid_thw.unsqueeze(0)
                
                # Forward through vision encoder
                vision_embeds = vision_model(batched_patches, batched_grid)
                
                # Remove batch dimension
                if len(vision_embeds.shape) == 3:
                    vision_embeds = vision_embeds[0]
                
                return vision_embeds
                
            except Exception as e:
                logger.warning(f"[Embedder] Could not use vision model: {e}")
        
        # Fallback: Project patches to hidden size
        # This is a simplified projection when we can't access the real vision model
        patch_dim = vision_patches.shape[1]
        
        # Create a simple linear projection
        projection = torch.nn.Linear(patch_dim, self.hidden_size, device=device)
        
        # Initialize with small values
        with torch.no_grad():
            projection.weight.normal_(0, 0.02)
            projection.bias.zero_()
        
        # Project patches
        vision_embeds = projection(vision_patches)
        
        # Add positional information based on grid
        t, h, w = grid_thw.tolist()
        
        # Create position embeddings
        position_embeds = self._create_position_embeddings(t, h, w, device)
        
        # Add to vision embeddings
        vision_embeds = vision_embeds + position_embeds[:seq_len]
        
        return vision_embeds
    
    def _create_position_embeddings(self, t: int, h: int, w: int, 
                                   device: torch.device) -> torch.Tensor:
        """
        Create 3D position embeddings for temporal and spatial dimensions.
        """
        seq_len = t * h * w
        
        # Create embeddings for each dimension
        pos_embeds = torch.zeros(seq_len, self.hidden_size, device=device)
        
        # Simple sinusoidal position encoding
        for i in range(seq_len):
            # Decode position
            frame_idx = i // (h * w)
            spatial_idx = i % (h * w)
            row_idx = spatial_idx // w
            col_idx = spatial_idx % w
            
            # Encode position into embedding
            # Temporal dimension
            pos_embeds[i, 0::6] = torch.sin(torch.tensor(frame_idx * 0.1))
            pos_embeds[i, 1::6] = torch.cos(torch.tensor(frame_idx * 0.1))
            
            # Height dimension
            pos_embeds[i, 2::6] = torch.sin(torch.tensor(row_idx * 0.01))
            pos_embeds[i, 3::6] = torch.cos(torch.tensor(row_idx * 0.01))
            
            # Width dimension
            pos_embeds[i, 4::6] = torch.sin(torch.tensor(col_idx * 0.01))
            pos_embeds[i, 5::6] = torch.cos(torch.tensor(col_idx * 0.01))
        
        return pos_embeds


def create_multiframe_tokens(text: str, frames: List[torch.Tensor],
                            vision_patches: torch.Tensor,
                            grid_thw: torch.Tensor,
                            debug: bool = False) -> Dict[str, Any]:
    """
    Main entry point for creating multi-frame tokens.
    
    Args:
        text: The prompt text
        frames: List of frame tensors
        vision_patches: Processed vision patches
        grid_thw: Grid dimensions
        debug: Enable debug logging
        
    Returns:
        Token dictionary for ComfyUI
    """
    tokenizer = QwenMultiFrameTokenizer()
    return tokenizer.create_multiframe_tokens(
        text, frames, vision_patches, grid_thw, debug
    )


# Export for use in nodes
__all__ = [
    'QwenMultiFrameTokenizer',
    'MultiFrameVisionEmbedder',
    'create_multiframe_tokens'
]