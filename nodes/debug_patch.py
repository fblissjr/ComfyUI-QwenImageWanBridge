"""
Debug patch for ComfyUI sampling pipeline - traces reference latents end-to-end
Apply this by running: import debug_patch; debug_patch.apply_debug_patches()
"""

import logging
import torch

logger = logging.getLogger(__name__)

def apply_debug_patches():
    """Apply comprehensive debugging patches to ComfyUI's sampling pipeline"""
    
    try:
        # Patch ComfyUI's model_base.py QwenImage.extra_conds
        import comfy.model_base
        original_extra_conds = comfy.model_base.QwenImage.extra_conds
        
        def debug_extra_conds(self, **kwargs):
            logger.info("=== COMFYUI MODEL_BASE EXTRA_CONDS TRACE ===")
            logger.info(f"STEP 2 - EXTRA_CONDS PROCESSING:")
            
            # Check for trace ID from our encoder
            if "trace_id" in kwargs:
                logger.info(f"  - Found trace ID: {kwargs['trace_id']}")
            
            # Log what we receive
            logger.info(f"  - Received kwargs keys: {list(kwargs.keys())}")
            if "reference_latents" in kwargs:
                ref_latents = kwargs["reference_latents"]
                logger.info(f"  - Reference latents type: {type(ref_latents)}")
                if isinstance(ref_latents, list):
                    logger.info(f"  - Reference latents count: {len(ref_latents)}")
                    for i, lat in enumerate(ref_latents[:3]):  # Only log first 3
                        if hasattr(lat, 'shape'):
                            logger.info(f"  - Ref latent {i}: {lat.shape}, {lat.dtype}, range=[{lat.min():.4f}, {lat.max():.4f}]")
            
            # Call original method
            result = original_extra_conds(self, **kwargs)
            
            # Log what we return
            logger.info(f"  - Returned keys: {list(result.keys())}")
            if "ref_latents" in result:
                logger.info(f"  - ref_latents type: {type(result['ref_latents'])}")
            logger.info("=== EXTRA_CONDS HANDOFF TO MODEL ===")
            
            return result
        
        comfy.model_base.QwenImage.extra_conds = debug_extra_conds
        logger.info("Applied QwenImage.extra_conds debug patch")
        
        # Patch ComfyUI's qwen_image model forward pass
        import comfy.ldm.qwen_image.model
        original_forward = comfy.ldm.qwen_image.model.QwenImageTransformer2DModel._forward
        
        def debug_forward(self, x, timesteps, context, attention_mask=None, guidance=None, ref_latents=None, transformer_options={}, control=None, **kwargs):
            logger.info("=== QWEN IMAGE MODEL FORWARD TRACE ===")
            logger.info(f"STEP 3 - MODEL FORWARD PASS:")
            logger.info(f"  - Input latent shape: {x.shape}")
            logger.info(f"  - Timesteps: {timesteps.shape if hasattr(timesteps, 'shape') else timesteps}")
            logger.info(f"  - Context shape: {context.shape if hasattr(context, 'shape') else type(context)}")
            
            if ref_latents is not None:
                logger.info(f"  - ref_latents type: {type(ref_latents)}")
                if hasattr(ref_latents, '__len__'):
                    logger.info(f"  - ref_latents count: {len(ref_latents)}")
                    for i, lat in enumerate(ref_latents[:3]):  # Only log first 3
                        if hasattr(lat, 'shape'):
                            logger.info(f"  - Model ref_latent {i}: {lat.shape}, range=[{lat.min():.4f}, {lat.max():.4f}]")
            else:
                logger.info(f"  - ref_latents: None - NO REFERENCE LATENTS RECEIVED!")
            
            # Call original forward
            result = original_forward(self, x, timesteps, context, attention_mask, guidance, ref_latents, transformer_options, control, **kwargs)
            
            logger.info(f"  - Output shape: {result.shape}")
            logger.info(f"  - Output range: [{result.min():.4f}, {result.max():.4f}]")
            logger.info("=== MODEL FORWARD COMPLETE ===")
            
            return result
        
        comfy.ldm.qwen_image.model.QwenImageTransformer2DModel._forward = debug_forward
        logger.info("Applied QwenImageTransformer2DModel._forward debug patch")
        
        logger.info("=== DEBUG PATCHES APPLIED SUCCESSFULLY ===")
        logger.info("Run your workflow now to see end-to-end tracing")
        
    except Exception as e:
        logger.error(f"Failed to apply debug patches: {e}")

def remove_debug_patches():
    """Remove debug patches (restart ComfyUI to fully reset)"""
    logger.info("Debug patches require ComfyUI restart to fully remove")

if __name__ == "__main__":
    apply_debug_patches()