"""
Qwen2.5-VL CLIP Wrapper for ComfyUI
Uses ComfyUI's internal Qwen loader but with DiffSynth-Studio templates
Includes all fixes from DiffSynth-Studio and DiffSynth-Engine
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, Tuple, Union
import folder_paths

logger = logging.getLogger(__name__)

# Try to import ComfyUI's utilities
try:
    import comfy.sd
    import comfy.model_management as mm
    COMFY_AVAILABLE = True
except ImportError:
    logger.warning("ComfyUI utilities not available")
    COMFY_AVAILABLE = False

# Apply RoPE position embedding fix from DiffSynth-Studio
def apply_rope_fix():
    """Monkey patch to fix batch processing with different image sizes"""
    try:
        import comfy.ldm.qwen_image.model as qwen_model
        
        # Check if the model has QwenEmbedRope (it might not exist)
        if not hasattr(qwen_model, 'QwenEmbedRope'):
            logger.info("QwenEmbedRope not found, skipping RoPE fix")
            return
            
        original_expand = qwen_model.QwenEmbedRope._expand_pos_freqs_if_needed
        
        def fixed_expand_pos_freqs(self, video_fhw, txt_seq_lens):
            # Apply fix from DiffSynth-Studio commit 8fcfa1d
            if isinstance(video_fhw, list):
                # Take max dimensions across batch instead of just first element
                video_fhw = tuple(max([i[j] for i in video_fhw]) for j in range(3))
            
            # Call original method with fixed video_fhw
            return original_expand(self, video_fhw, txt_seq_lens)
        
        qwen_model.QwenEmbedRope._expand_pos_freqs_if_needed = fixed_expand_pos_freqs
        logger.info("Applied RoPE position embedding fix for batch processing")
        
    except Exception as e:
        logger.warning(f"Could not apply RoPE fix: {e}")

# Apply fix on module load
apply_rope_fix()

class QwenVLCLIPLoader:
    """
    Load Qwen2.5-VL using ComfyUI's internal CLIP loader
    This ensures compatibility with the diffusion pipeline
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get models from text_encoders folder
        models = folder_paths.get_filename_list("text_encoders")
        # Filter for Qwen models
        qwen_models = [m for m in models if "qwen" in m.lower()]
        if not qwen_models:
            qwen_models = ["qwen_2.5_vl_7b.safetensors"]
        
        return {
            "required": {
                "model_name": (qwen_models, {
                    "tooltip": "Qwen2.5-VL model from 'ComfyUI/models/text_encoders'"
                }),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "QwenImage/Loaders"
    TITLE = "Qwen2.5-VL CLIP Loader"
    DESCRIPTION = "Load Qwen2.5-VL as CLIP for ComfyUI compatibility"
    
    def load_clip(self, model_name: str) -> Tuple[Any]:
        """Load Qwen2.5-VL using ComfyUI's CLIP loader"""
        
        if not COMFY_AVAILABLE:
            raise RuntimeError("ComfyUI not available")
        
        # Get full path
        model_path = folder_paths.get_full_path("text_encoders", model_name)
        logger.info(f"Loading Qwen2.5-VL from: {model_path}")
        
        # Load using ComfyUI's CLIP loader with qwen_image type
        clip = comfy.sd.load_clip(
            ckpt_paths=[model_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.QWEN_IMAGE
        )
        
        logger.info("Successfully loaded Qwen2.5-VL as CLIP")
        return (clip,)


class QwenVLTextEncoder:
    """
    Enhanced text encoder for Qwen2.5-VL with all DiffSynth fixes
    Uses ComfyUI's internal CLIP infrastructure for compatibility
    """
    
    # DiffSynth-Studio's exact resolution list for Qwen
    QWEN_RESOLUTIONS = [
        (256, 256), (256, 512), (256, 768), (256, 1024), (256, 1280), (256, 1536), (256, 1792),
        (512, 256), (512, 512), (512, 768), (512, 1024), (512, 1280), (512, 1536), (512, 1792),
        (768, 256), (768, 512), (768, 768), (768, 1024), (768, 1280), (768, 1536),
        (1024, 256), (1024, 512), (1024, 768), (1024, 1024), (1024, 1280), (1024, 1536),
        (1280, 256), (1280, 512), (1280, 768), (1280, 1024), (1280, 1280),
        (1536, 256), (1536, 512), (1536, 768), (1536, 1024),
        (1792, 256), (1792, 512)
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape",
                    "tooltip": "Your prompt. Just write what you want, templates are applied automatically."
                }),
                "mode": (["text_to_image", "image_edit"], {
                    "default": "image_edit",
                    "tooltip": "text_to_image: Generate from scratch | image_edit: Modify existing image"
                }),
            },
            "optional": {
                "edit_image": ("IMAGE", {
                    "tooltip": "Image to edit/reference. ALWAYS provide this for image_edit mode!"
                }),
                "multi_reference": ("QWEN_MULTI_REF", {
                    "tooltip": "Multiple reference images from Multi-Reference Handler (overrides edit_image)"
                }),
                "vae": ("VAE", {
                    "tooltip": "ALWAYS connect VAE! Encodes reference image for vision guidance."
                }),
                "use_custom_system_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "ON: Text from Template Builder (already formatted) | OFF: Apply default Qwen formatting"
                }),
                "token_removal": (["auto", "diffsynth", "none"], {
                    "default": "auto",
                    "tooltip": "Keep 'auto' unless you know why you need others. Auto=smart, diffsynth=exact compatibility, none=keep all"
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Shows detailed processing info in console. Turn on if things aren't working."
                }),
                "optimize_resolution": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "ON: Snap to best Qwen resolution (better quality) | OFF: Scale to 1M pixels"
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "QwenImage/Encoding"
    TITLE = "Qwen2.5-VL Text Encoder"
    DESCRIPTION = """
Encodes text and images for Qwen Image generation.

KEY DECISION: What goes into KSampler.latent_image?
• VAE Encode → KSampler = Preserve structure (denoise 0.3-0.7)
• Empty Latent → KSampler = Full reimagining (denoise 0.9-1.0)

ALWAYS connect VAE to this node for reference latents!
"""
    
    def get_optimal_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Find the nearest Qwen-supported resolution preserving aspect ratio"""
        target_pixels = width * height
        aspect_ratio = width / height
        
        # Find closest resolution by both pixel count AND aspect ratio
        best_res = min(
            self.QWEN_RESOLUTIONS,
            key=lambda r: abs(r[0] * r[1] - target_pixels) * 0.5 + 
                         abs((r[0] / r[1]) - aspect_ratio) * target_pixels * 0.5
        )
        
        return best_res
    
    # Template styles moved to Template Builder node - removed to avoid duplication
    
    def encode(self, clip, text: str, mode: str = "text_to_image",
              edit_image: Optional[torch.Tensor] = None, multi_reference: Optional[Dict] = None,
              vae=None, use_custom_system_prompt: bool = False, token_removal: str = "auto",
              debug_mode: bool = False,
              optimize_resolution: bool = False) -> Tuple[Any]:
        """
        Enhanced encoding with all DiffSynth features
        
        Args:
            clip: ComfyUI CLIP model
            text: Input prompt text
            mode: Either "text_to_image" or "image_edit"
            edit_image: Optional image tensor for edit mode
            vae: Optional VAE for reference latents
            use_custom_system_prompt: Whether text is pre-formatted from Template Builder
            optimize_resolution: Auto-optimize to Qwen resolution
            token_removal: How to remove template tokens
            debug_mode: Enable debug logging
        """
        
        images = []
        ref_latent = None  # Initialize here so it's in scope later
        original_text = text  # Store original for custom prompting
        
        # Handle multi-reference input if provided (overrides edit_image)
        if multi_reference is not None:
            if debug_mode:
                logger.info(f"[DEBUG] Processing multi-reference with {multi_reference['count']} images")
                logger.info(f"[DEBUG] Multi-ref method: {multi_reference['method']}")
            
            multi_images = multi_reference["images"]
            multi_method = multi_reference["method"]
            multi_weights = multi_reference.get("weights", [1.0] * len(multi_images))
            
            # Determine reference method from multi-ref
            reference_method = multi_method if multi_method in ["index", "offset"] else "standard"
            
            # Process based on method
            if multi_method == "index":
                # For index method, use first image as primary
                # TODO: Full implementation would process all images separately
                edit_image = multi_images[0]
                if debug_mode:
                    logger.info(f"[DEBUG] Index method: using first of {len(multi_images)} images as primary")
                    
            elif multi_method == "offset":
                # Weighted average of images
                edit_image = torch.zeros_like(multi_images[0])
                for img, weight in zip(multi_images, multi_weights):
                    edit_image = edit_image + img * weight
                if debug_mode:
                    logger.info(f"[DEBUG] Offset method: blended {len(multi_images)} images with weights {multi_weights}")
                    
            elif multi_method == "concat":
                # Concatenate horizontally
                edit_image = torch.cat(multi_images, dim=2)
                if debug_mode:
                    logger.info(f"[DEBUG] Concat method: combined {len(multi_images)} images horizontally")
                    
            elif multi_method == "grid":
                # Create 2x2 grid
                if len(multi_images) == 1:
                    edit_image = multi_images[0]
                elif len(multi_images) == 2:
                    edit_image = torch.cat(multi_images, dim=1)  # Stack vertically
                else:
                    row1 = torch.cat(multi_images[:2], dim=2)
                    if len(multi_images) >= 4:
                        row2 = torch.cat(multi_images[2:4], dim=2)
                    else:  # 3 images
                        row2 = torch.cat([multi_images[2], torch.zeros_like(multi_images[2])], dim=2)
                    edit_image = torch.cat([row1, row2], dim=1)
                if debug_mode:
                    logger.info(f"[DEBUG] Grid method: arranged {len(multi_images)} images in grid")
        
        # Set default reference method if not set by multi-ref
        if 'reference_method' not in locals():
            reference_method = "standard"
        
        # Prepare image if in edit mode - following ComfyUI's TextEncodeQwenImageEdit pattern
        if mode == "image_edit" and edit_image is not None:
            # Process image tensor like the official node does
            import math
            import comfy.utils
            
            if debug_mode:
                logger.info(f"[DEBUG] Input image shape: {edit_image.shape}")
                logger.info(f"[DEBUG] Input image dtype: {edit_image.dtype}")
                logger.info(f"[DEBUG] Input image min/max: {edit_image.min():.4f}/{edit_image.max():.4f}")
            
            # ComfyUI IMAGE is [B, H, W, C], we need to process it
            samples = edit_image.movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
            
            if debug_mode:
                logger.info(f"[DEBUG] After movedim shape: {samples.shape}")
            
            # Determine target resolution
            if optimize_resolution:
                # Use optimal Qwen resolution
                opt_w, opt_h = self.get_optimal_resolution(samples.shape[3], samples.shape[2])
                width, height = opt_w, opt_h
                if debug_mode:
                    logger.info(f"[DEBUG] Using optimal Qwen resolution: {width}x{height}")
            else:
                # Scale to target resolution (1024x1024 total pixels like official)
                total = int(1024 * 1024)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
            
            if debug_mode:
                logger.info(f"[DEBUG] Scaling from {samples.shape[3]}x{samples.shape[2]} to {width}x{height}")
            
            # Resize using ComfyUI's common_upscale
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)  # [B, C, H, W] -> [B, H, W, C]
            
            if debug_mode:
                logger.info(f"[DEBUG] After resize shape: {image.shape}")
                logger.info(f"[DEBUG] After resize min/max: {image.min():.4f}/{image.max():.4f}")
            
            # Extract RGB channels (drop alpha if present)
            images = [image[:, :, :, :3]]
            
            if debug_mode:
                logger.info(f"[DEBUG] Final image shape for tokenizer: {images[0].shape}")
                logger.info(f"[DEBUG] Final image dtype: {images[0].dtype}")
                logger.info(f"[DEBUG] Final image min/max: {images[0].min():.4f}/{images[0].max():.4f}")
            
            # Add reference latents if VAE provided (like official node)
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])
                if debug_mode:
                    logger.info(f"[DEBUG] Encoded reference latent shape: {ref_latent.shape}")
                    logger.info(f"[DEBUG] Reference latent min/max: {ref_latent.min():.4f}/{ref_latent.max():.4f}")
        
        # Handle template application
        if use_custom_system_prompt:
            # User wants to use custom prompting - pass text as-is
            if debug_mode:
                logger.info(f"[DEBUG] Using custom formatted text from Template Builder")
        else:
            # For backward compatibility with token_removal="diffsynth"
            if token_removal == "diffsynth" and not use_custom_system_prompt:
                # Use exact DiffSynth templates for compatibility
                if mode == "text_to_image":
                    template = (
                        "<|im_start|>system\n"
                        "Describe the image by detailing the color, shape, size, texture, "
                        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                        "<|im_start|>user\n{}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                    text = template.format(original_text)
                elif mode == "image_edit" and edit_image is not None:
                    template = (
                        "<|im_start|>system\n"
                        "Describe the key features of the input image (color, shape, size, texture, objects, background), "
                        "then explain how the user's text instruction should alter or modify the image. "
                        "Generate a new image that meets the user's requirements while maintaining consistency "
                        "with the original input where appropriate.<|im_end|>\n"
                        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                    text = template.format(original_text)
                if debug_mode:
                    logger.info(f"[DEBUG] Applied DiffSynth template for {mode} (backward compatibility)")
            else:
                # Apply default Qwen formatting
                if mode == "text_to_image":
                    template = (
                        "<|im_start|>system\n"
                        "Describe the image by detailing the color, shape, size, texture, "
                        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                        "<|im_start|>user\n{}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                else:  # image_edit
                    template = (
                        "<|im_start|>system\n"
                        "Describe the key features of the input image (color, shape, size, texture, objects, background), "
                        "then explain how the user's text instruction should alter or modify the image. "
                        "Generate a new image that meets the user's requirements while maintaining consistency "
                        "with the original input where appropriate.<|im_end|>\n"
                        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                text = template.format(original_text)
                if debug_mode:
                    logger.info(f"[DEBUG] Applied default Qwen formatting for {mode}")
        
        # Tokenize
        if debug_mode:
            logger.info(f"[DEBUG] Tokenizing with text: '{text[:50]}...' and {len(images)} images")
        
        if images:
            tokens = clip.tokenize(text, images=images)
        else:
            tokens = clip.tokenize(text)
        
        # Debug token info
        if debug_mode and isinstance(tokens, dict):
            for key in tokens:
                if isinstance(tokens[key], list) and len(tokens[key]) > 0:
                    logger.info(f"[DEBUG] Token key '{key}' has {len(tokens[key][0])} tokens")
        
        # Handle token removal based on mode
        if token_removal == "diffsynth" and not use_custom_system_prompt:
            # DiffSynth-style fixed removal
            drop_count = 34 if mode == "text_to_image" else 64
            
            # Manually remove tokens before encoding
            for key in tokens:
                if isinstance(tokens[key], list) and len(tokens[key]) > 0:
                    for i in range(len(tokens[key])):
                        token_list = tokens[key][i]
                        if len(token_list) > drop_count:
                            tokens[key][i] = token_list[drop_count:]
                            if debug_mode:
                                logger.info(f"[DEBUG] Dropped first {drop_count} tokens (DiffSynth style)")
        
        elif token_removal == "none":
            # Keep all tokens
            if debug_mode:
                logger.info(f"[DEBUG] Keeping all tokens (no removal)")
        
        # For "auto" mode, ComfyUI's encode_from_tokens_scheduled handles it
        
        # Encode tokens using ComfyUI's method
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # Add reference latents if we have them (like official node)
        if ref_latent is not None:
            import node_helpers
            ref_data = {"reference_latents": [ref_latent]}
            
            # Add reference method for proper handling (Flux-style)
            if reference_method != "standard":
                ref_data["reference_latents_method"] = reference_method
            
            conditioning = node_helpers.conditioning_set_values(conditioning, ref_data, append=True)
            if debug_mode:
                logger.info(f"[DEBUG] Added reference latents to conditioning with method: {reference_method}")
        
        # Debug conditioning
        if debug_mode and isinstance(conditioning, list) and len(conditioning) > 0:
            cond_tensor = conditioning[0][0]
            logger.info(f"[DEBUG] Conditioning shape: {cond_tensor.shape}")
            logger.info(f"[DEBUG] Conditioning dtype: {cond_tensor.dtype}")
            logger.info(f"[DEBUG] Conditioning min/max: {cond_tensor.min():.4f}/{cond_tensor.max():.4f}")
            logger.info(f"[DEBUG] Conditioning mean/std: {cond_tensor.mean():.4f}/{cond_tensor.std():.4f}")
            
            # Check if there's metadata
            if len(conditioning[0]) > 1:
                metadata = conditioning[0][1]
                logger.info(f"[DEBUG] Conditioning metadata keys: {metadata.keys() if isinstance(metadata, dict) else 'Not a dict'}")
                if isinstance(metadata, dict) and 'reference_latents' in metadata:
                    logger.info(f"[DEBUG] Has reference_latents: {len(metadata['reference_latents'])} items")
        
        if debug_mode:
            logger.info(f"[DEBUG] Encoded text in {mode} mode")
        
        return (conditioning,)


class QwenLowresFixNode:
    """
    Makes your image BETTER with two-stage refinement.
    Stage 1: Generate at current size
    Stage 2: Upscale and polish details
    
    Connect AFTER your first KSampler for quality boost!
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Import KSampler list here to avoid import issues
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except:
            samplers = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral"]
            schedulers = ["normal", "karras", "exponential", "simple", "ddim_uniform"]
            
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "How much to refine? 0.3-0.5=subtle polish | 0.5-0.7=moderate | 0.7+=heavy changes"
                }),
                "upscale_factor": ("FLOAT", {
                    "default": 1.5, 
                    "min": 1.0, 
                    "max": 4.0, 
                    "step": 0.1,
                    "tooltip": "How much bigger? 1.5x is usually perfect. 2x+ needs more VRAM."
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"
    CATEGORY = "QwenImage/Refinement"
    TITLE = "Qwen Lowres Fix"
    DESCRIPTION = "Two-stage refinement for higher quality (DiffSynth-Studio method)"
    
    def process(self, model, positive, negative, latent, vae, seed, steps,
                cfg, sampler_name, scheduler, denoise, upscale_factor):
        """
        Implement two-stage generation:
        1. First pass at current resolution with full denoise
        2. Upscale and refine with partial denoise
        """
        import comfy.samplers
        import comfy.utils
        
        # Stage 1: Full generation at current resolution
        sampler = comfy.samplers.KSampler()
        stage1_samples = sampler.sample(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.0  # Full denoise for initial generation
        )
        
        # Decode to image space
        stage1_latent = stage1_samples[0]["samples"]
        decoded = vae.decode(stage1_latent)
        
        # Upscale the image
        # ComfyUI images are [B, H, W, C]
        h, w = decoded.shape[1], decoded.shape[2]
        new_h = int(h * upscale_factor)
        new_w = int(w * upscale_factor)
        
        # Convert to [B, C, H, W] for upscaling
        decoded_chw = decoded.movedim(-1, 1)
        upscaled = comfy.utils.common_upscale(
            decoded_chw, new_w, new_h, "bicubic", "disabled"
        )
        # Convert back to [B, H, W, C]
        upscaled_hwc = upscaled.movedim(1, -1)
        
        # Encode back to latent space
        stage2_latent = vae.encode(upscaled_hwc[:, :, :, :3])  # RGB only
        
        # Stage 2: Refinement with partial denoise
        refined_samples = sampler.sample(
            model=model,
            seed=seed + 1,  # Different seed for variation
            steps=max(steps // 2, 10),  # Fewer steps for refinement
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image={"samples": stage2_latent},
            denoise=denoise  # Partial denoise for refinement
        )
        
        return refined_samples


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenVLCLIPLoader": QwenVLCLIPLoader,
    "QwenVLTextEncoder": QwenVLTextEncoder,
    "QwenLowresFixNode": QwenLowresFixNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLCLIPLoader": "Qwen2.5-VL CLIP Loader",
    "QwenVLTextEncoder": "Qwen2.5-VL Text Encoder (Enhanced)",
    "QwenLowresFixNode": "Qwen Lowres Fix (Two-Stage)",
}