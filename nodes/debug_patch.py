"""
Debug patch for ComfyUI sampling pipeline - traces reference latents end-to-end
Apply this by running: import debug_patch; debug_patch.apply_debug_patches()
"""

import logging
import torch
import os
import time
import traceback

logger = logging.getLogger(__name__)

# Global debug flag - can be set via environment variable or directly
DEBUG_VERBOSE = os.environ.get('QWEN_DEBUG_VERBOSE', '').lower() in ('true', '1', 'yes')

def set_debug_verbose(enabled: bool):
    """Enable or disable verbose debug output"""
    global DEBUG_VERBOSE
    DEBUG_VERBOSE = enabled

def apply_debug_patches():
    """Apply comprehensive debugging patches to ComfyUI's sampling pipeline"""
    
    try:
        # Patch ComfyUI's model_base.py QwenImage.extra_conds
        import comfy.model_base
        original_extra_conds = comfy.model_base.QwenImage.extra_conds
        
        def debug_extra_conds(self, **kwargs):
            if DEBUG_VERBOSE:
                logger.info("\n" + "="*60)
                logger.info("COMFYUI MODEL_BASE EXTRA_CONDS TRACE")
                logger.info("="*60)
                logger.info(f"[STEP 2] Processing extra conditions for model")
                logger.info(f"Timestamp: {time.strftime('%H:%M:%S')}")

                # Check for trace ID from our encoder
                if "trace_id" in kwargs:
                    logger.info(f"  ✓ Trace ID: {kwargs['trace_id']}")

                # Log all received parameters
                logger.info(f"\n[Received Parameters]")
                logger.info(f"  Keys: {list(kwargs.keys())}")

                # Detailed conditioning info
                if "c" in kwargs:
                    cond = kwargs["c"]
                    if isinstance(cond, list):
                        logger.info(f"  Conditioning: list of {len(cond)} items")
                        for i, c in enumerate(cond[:2]):
                            if isinstance(c, torch.Tensor):
                                logger.info(f"    - Cond {i}: shape={c.shape}, dtype={c.dtype}")

                # Reference latents detailed logging
                if "reference_latents" in kwargs:
                    ref_latents = kwargs["reference_latents"]
                    logger.info(f"\n[Reference Latents]")
                    logger.info(f"  Type: {type(ref_latents).__name__}")
                    if isinstance(ref_latents, list):
                        logger.info(f"  Count: {len(ref_latents)} latents")
                        for i, lat in enumerate(ref_latents):
                            if hasattr(lat, 'shape'):
                                logger.info(f"  Latent {i+1}:")
                                logger.info(f"    - Shape: {lat.shape}")
                                logger.info(f"    - Dtype: {lat.dtype}")
                                logger.info(f"    - Device: {lat.device}")
                                logger.info(f"    - Range: [{lat.min().item():.4f}, {lat.max().item():.4f}]")
                                logger.info(f"    - Mean: {lat.mean().item():.4f}, Std: {lat.std().item():.4f}")
                else:
                    logger.info(f"\n[Reference Latents]")
                    logger.info(f"  ⚠ NO REFERENCE LATENTS PROVIDED")

                # Log transformer options if present
                if "transformer_options" in kwargs:
                    opts = kwargs["transformer_options"]
                    if opts:
                        logger.info(f"\n[Transformer Options]")
                        logger.info(f"  Keys: {list(opts.keys()) if isinstance(opts, dict) else 'Not a dict'}")

            # Call original method
            start_time = time.perf_counter() if DEBUG_VERBOSE else 0
            result = original_extra_conds(self, **kwargs)
            elapsed = (time.perf_counter() - start_time) * 1000 if DEBUG_VERBOSE else 0

            if DEBUG_VERBOSE:
                # Log what we return
                logger.info(f"\n[Output to Model]")
                logger.info(f"  Keys: {list(result.keys())}")
                if "ref_latents" in result:
                    ref_out = result["ref_latents"]
                    logger.info(f"  ref_latents:")
                    logger.info(f"    - Type: {type(ref_out).__name__}")
                    if isinstance(ref_out, list):
                        logger.info(f"    - Count: {len(ref_out)}")
                logger.info(f"\n[Performance]")
                logger.info(f"  Processing time: {elapsed:.2f}ms")
                logger.info("="*60 + "\n")
            
            return result
        
        comfy.model_base.QwenImage.extra_conds = debug_extra_conds
        if DEBUG_VERBOSE:
            logger.info("Applied QwenImage.extra_conds debug patch (VERBOSE MODE)")
        else:
            logger.info("Applied QwenImage.extra_conds debug patch (SILENT MODE)")
        
        # Patch ComfyUI's qwen_image model forward pass
        import comfy.ldm.qwen_image.model
        original_forward = comfy.ldm.qwen_image.model.QwenImageTransformer2DModel._forward
        
        def debug_forward(self, x, timesteps, context, attention_mask=None, guidance=None, ref_latents=None, transformer_options={}, control=None, **kwargs):
            if DEBUG_VERBOSE:
                logger.info("\n" + "="*60)
                logger.info("QWEN IMAGE MODEL FORWARD PASS")
                logger.info("="*60)
                logger.info(f"[STEP 3] Executing model forward pass")
                logger.info(f"Timestamp: {time.strftime('%H:%M:%S')}")

                # Input details
                logger.info(f"\n[Input Tensors]")
                logger.info(f"  Latent (x):")
                logger.info(f"    - Shape: {x.shape}")
                logger.info(f"    - Device: {x.device}")
                logger.info(f"    - Range: [{x.min().item():.4f}, {x.max().item():.4f}]")

                logger.info(f"  Timesteps:")
                if hasattr(timesteps, 'shape'):
                    logger.info(f"    - Shape: {timesteps.shape}")
                    logger.info(f"    - Values: {timesteps.tolist() if timesteps.numel() < 10 else f'{timesteps[0].item():.2f}...'}")
                else:
                    logger.info(f"    - Value: {timesteps}")

                logger.info(f"  Context:")
                if hasattr(context, 'shape'):
                    logger.info(f"    - Shape: {context.shape}")
                    logger.info(f"    - Device: {context.device}")
                else:
                    logger.info(f"    - Type: {type(context).__name__}")

                # Attention mask info
                if attention_mask is not None:
                    logger.info(f"  Attention Mask:")
                    logger.info(f"    - Shape: {attention_mask.shape if hasattr(attention_mask, 'shape') else 'N/A'}")
                    logger.info(f"    - Unique values: {torch.unique(attention_mask).tolist() if hasattr(attention_mask, 'unique') else 'N/A'}")

                # Guidance info
                if guidance is not None:
                    logger.info(f"  Guidance: {guidance}")

                # Reference latents detailed info
                logger.info(f"\n[Reference Latents in Model]")
                if ref_latents is not None:
                    logger.info(f"  Type: {type(ref_latents).__name__}")
                    if hasattr(ref_latents, '__len__'):
                        logger.info(f"  Count: {len(ref_latents)} latents")
                        for i, lat in enumerate(ref_latents):
                            if hasattr(lat, 'shape'):
                                logger.info(f"  Latent {i+1}:")
                                logger.info(f"    - Shape: {lat.shape}")
                                logger.info(f"    - Device: {lat.device}")
                                logger.info(f"    - Range: [{lat.min().item():.4f}, {lat.max().item():.4f}]")
                                logger.info(f"    - Mean: {lat.mean().item():.4f}, Std: {lat.std().item():.4f}")
                                # Check for NaN or Inf
                                if torch.isnan(lat).any():
                                    logger.warning(f"    ⚠ Contains NaN values!")
                                if torch.isinf(lat).any():
                                    logger.warning(f"    ⚠ Contains Inf values!")
                else:
                    logger.warning(f"  ⚠ NO REFERENCE LATENTS RECEIVED BY MODEL!")
                    logger.info(f"  This means image editing won't use reference images")

                # Transformer options
                if transformer_options:
                    logger.info(f"\n[Transformer Options]")
                    logger.info(f"  Keys: {list(transformer_options.keys())}")
                    if 'block_modifiers' in transformer_options:
                        logger.info(f"  Block modifiers present: {len(transformer_options['block_modifiers'])}")

            # Call original forward
            start_time = time.perf_counter() if DEBUG_VERBOSE else 0
            result = original_forward(self, x, timesteps, context, attention_mask, guidance, ref_latents, transformer_options, control, **kwargs)
            elapsed = (time.perf_counter() - start_time) * 1000 if DEBUG_VERBOSE else 0

            if DEBUG_VERBOSE:
                logger.info(f"\n[Output]")
                logger.info(f"  Shape: {result.shape}")
                logger.info(f"  Device: {result.device}")
                logger.info(f"  Range: [{result.min().item():.4f}, {result.max().item():.4f}]")
                logger.info(f"  Mean: {result.mean().item():.4f}, Std: {result.std().item():.4f}")

                # Check output health
                if torch.isnan(result).any():
                    logger.error(f"  ⚠ Output contains NaN values!")
                if torch.isinf(result).any():
                    logger.error(f"  ⚠ Output contains Inf values!")

                logger.info(f"\n[Performance]")
                logger.info(f"  Forward pass time: {elapsed:.2f}ms")
                logger.info("="*60 + "\n")
            
            return result
        
        comfy.ldm.qwen_image.model.QwenImageTransformer2DModel._forward = debug_forward
        if DEBUG_VERBOSE:
            logger.info("Applied QwenImageTransformer2DModel._forward debug patch (VERBOSE MODE)")
            logger.info("=== DEBUG PATCHES APPLIED SUCCESSFULLY ===")
            logger.info("Run your workflow now to see end-to-end tracing")
        else:
            logger.info("Applied debug patches (SILENT MODE - set QWEN_DEBUG_VERBOSE=true for output)")
        
    except Exception as e:
        logger.error(f"Failed to apply debug patches: {e}")

def remove_debug_patches():
    """Remove debug patches (restart ComfyUI to fully reset)"""
    logger.info("Debug patches require ComfyUI restart to fully remove")

if __name__ == "__main__":
    apply_debug_patches()