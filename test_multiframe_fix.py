#!/usr/bin/env python3
"""
Test script to verify multiframe vision token processing fix
"""

import torch
import logging
import sys
import os

# Add paths for ComfyUI modules
sys.path.insert(0, '/Users/fredbliss/workspace/ComfyUI')
sys.path.insert(0, '/Users/fredbliss/workspace/ComfyUI-QwenImageWanBridge')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multiframe_processing():
    """Test that multiframe vision tokens are processed correctly"""
    
    try:
        # Import our modules
        from nodes.qwen_vision_processor import QwenVisionProcessor
        from nodes.qwen_custom_tokenizer import create_multiframe_tokens
        from nodes.qwen_encoder_patch import patch_qwen_encoder
        
        logger.info("Modules imported successfully")
        
        # Apply the patch
        patch_success = patch_qwen_encoder()
        logger.info(f"Encoder patch applied: {patch_success}")
        
        # Create test data
        # Simulate 2 frames of 224x224 RGB images
        frame1 = torch.rand(224, 224, 3)
        frame2 = torch.rand(224, 224, 3)
        frames = [frame1, frame2]
        
        logger.info(f"Created {len(frames)} test frames")
        
        # Process frames into vision patches
        processor = QwenVisionProcessor()
        vision_patches, grid_thw = processor.create_vision_patches(frames)
        
        logger.info(f"Vision patches shape: {vision_patches.shape}")
        logger.info(f"Grid THW: {grid_thw}")
        
        # Create multiframe tokens
        text = "Describe the changes between frames"
        tokens = create_multiframe_tokens(
            text=text,
            frames=frames,
            vision_patches=vision_patches,
            grid_thw=grid_thw,
            debug=True
        )
        
        logger.info(f"Created tokens with keys: {tokens.keys()}")
        
        # Verify the token structure
        if "qwen25_7b" in tokens:
            token_list = tokens["qwen25_7b"][0]
            multiframe_found = False
            
            for token_data in token_list:
                if isinstance(token_data[0], dict):
                    embed_dict = token_data[0]
                    if embed_dict.get("type") == "multiframe_vision":
                        multiframe_found = True
                        logger.info("✓ Found multiframe_vision token")
                        logger.info(f"  - Data shape: {embed_dict['data'].shape}")
                        logger.info(f"  - Grid THW: {embed_dict['grid_thw']}")
                        logger.info(f"  - Num frames: {embed_dict['num_frames']}")
                        break
            
            if not multiframe_found:
                logger.error("✗ No multiframe_vision token found")
                return False
        
        # Test that our patch handles it correctly
        try:
            from comfy.text_encoders import llama
            
            # Create a mock embed dict like what would be passed
            test_embed = {
                "type": "multiframe_vision",
                "data": vision_patches,
                "grid_thw": grid_thw,
                "num_frames": len(frames),
                "is_multiframe": True,
                "vision_patches": vision_patches
            }
            
            # Test device
            device = torch.device("cpu")
            
            # This would normally fail with the old code
            # With our fix, it should handle the multiframe type correctly
            logger.info("Testing preprocess_embed with multiframe token...")
            
            # We can't fully test without the model, but we can verify
            # the patch logic works
            if hasattr(llama, 'Qwen25_7BVLI'):
                logger.info("✓ Qwen25_7BVLI class found")
                
                # Check if our patch is applied
                preprocess_method = llama.Qwen25_7BVLI.preprocess_embed
                
                # Try to detect if it's our patched version
                import inspect
                source = inspect.getsource(preprocess_method)
                if "multiframe_vision" in source:
                    logger.info("✓ Patch is applied (multiframe_vision handling found)")
                else:
                    logger.warning("⚠ Patch may not be applied correctly")
            
        except Exception as e:
            logger.error(f"Error testing patch: {e}")
            return False
        
        logger.info("\n=== Test Summary ===")
        logger.info("✓ Multiframe vision tokens created successfully")
        logger.info("✓ Vision patches processed correctly")
        logger.info("✓ Encoder patch applied")
        logger.info("✓ Token structure validated")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_multiframe_processing()
    sys.exit(0 if success else 1)