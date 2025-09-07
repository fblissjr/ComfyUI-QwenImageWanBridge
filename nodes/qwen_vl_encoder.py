"""
Qwen2.5-VL CLIP Wrapper for ComfyUI
Uses ComfyUI's internal Qwen loader but with DiffSynth-Studio templates
Includes all fixes from DiffSynth-Studio and DiffSynth-Engine
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, Tuple, Union, List
import folder_paths

# Import our processor and custom tokenizer for proper multi-frame support
try:
    from .qwen_processor import Qwen2VLProcessor
    PROCESSOR_AVAILABLE = True
except ImportError:
    PROCESSOR_AVAILABLE = False

try:
    from .qwen_custom_tokenizer import QwenMultiFrameTokenizer, MultiFrameVisionEmbedder
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Try to import ComfyUI's utilities
try:
    import comfy.sd
    import comfy.model_management as mm
    import node_helpers
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
    Text encoder for Qwen2.5-VL with all DiffSynth fixes
    Uses ComfyUI's internal CLIP infrastructure for compatibility
    """

    # Resolution list combining DiffSynth-Studio resolutions with modern aspect ratios
    QWEN_RESOLUTIONS = [
        # Square resolutions
        (1024, 1024), (1328, 1328),
        
        # Common landscape ratios (optimized for quality)
        (1328, 800), (1456, 720), (1584, 1056), (1920, 1080),  # 16:9
        (2048, 1024), (1344, 768), (1536, 640),
        
        # Common portrait ratios
        (800, 1328), (720, 1456), (1056, 1584), (1080, 1920),  # 9:16
        (1024, 2048), (768, 1344), (640, 1536),
        
        # Original DiffSynth-Studio resolutions for compatibility
        (672, 1568), (688, 1504), (752, 1392), (832, 1248),
        (880, 1184), (944, 1104), (1104, 944), (1184, 880),
        (1248, 832), (1392, 752), (1504, 688), (1568, 672),
        
        # Smaller resolutions for low VRAM
        (512, 512), (768, 768), (512, 768), (768, 512),
        (1024, 768), (768, 1024), (1024, 512), (512, 1024)
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
                    "tooltip": "Image to edit/reference. Can be a single image or a composite canvas from the Multi-Reference Composer."
                }),
                "context_image": ("IMAGE", {
                    "tooltip": "Context/control image (ControlNet-style). Processed without vision tokens."
                }),
                "vae": ("VAE", {
                    "tooltip": "ALWAYS connect VAE! Encodes images for guidance."
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



    def encode(self, clip, text: str, mode: str = "text_to_image",
              edit_image: Optional[torch.Tensor] = None, context_image: Optional[torch.Tensor] = None,
              vae=None, use_custom_system_prompt: bool = False,
              token_removal: str = "auto", debug_mode: bool = False) -> Tuple[Any]:

        images_for_tokenizer = []
        ref_latent = None
        context_latent = None
        original_text = text
        dual_encoding_data = None
        
        # No resolution controls needed - images will be processed optimally

        # Prepare edit_image if in edit mode
        if mode == "image_edit" and edit_image is not None:
            if debug_mode:
                logger.info(f"[Encoder] Input image/canvas shape: {edit_image.shape}")

            # Use input image as-is for vision processing (no resizing needed)
            image = edit_image
            images_for_tokenizer = [image[:, :, :, :3]]

            # DUAL ENCODING: Process through both semantic and reconstructive paths
            if vae is not None:
                # Reconstructive path - standard VAE encoding
                ref_latent = vae.encode(image[:, :, :, :3])
                
                # DUAL ENCODING: Semantic path using native-level processing
                try:
                    from .qwen_vision_processor import QwenVisionProcessor
                    from .qwen_processor import Qwen2VLProcessor
                    from .qwen_custom_tokenizer import MultiFrameVisionEmbedder
                    
                    # Create advanced vision features (native-quality processing)
                    vision_processor = QwenVisionProcessor()
                    qwen_processor = Qwen2VLProcessor() 
                    embedder = MultiFrameVisionEmbedder()
                    
                    # Process image through semantic vision pipeline
                    image_list = [image[0]]  # Remove batch dimension for processor
                    semantic_patches, semantic_grid = vision_processor.create_vision_patches(image_list)
                    
                    # Create semantic embeddings (paper's semantic path)
                    semantic_embeddings = embedder.embed_vision_patches(
                        semantic_patches, semantic_grid, vision_model=None
                    )
                    
                    # Store dual encoding data for conditioning fusion (paper architecture)
                    dual_encoding_data = {
                        "semantic_embeddings": semantic_embeddings,  # High-level understanding
                        "semantic_patches": semantic_patches,
                        "semantic_grid": semantic_grid,
                        "reconstructive_latent": ref_latent,  # Low-level structure
                        "fusion_method": "mmdit_compatible",  # Paper's MMDiT fusion
                        "has_dual_encoding": True
                    }
                    
                    if debug_mode:
                        logger.info(f"[Encoder] Dual encoding - Semantic patches: {semantic_patches.shape}")
                        logger.info(f"[Encoder] Dual encoding - Reconstructive latent: {ref_latent.shape}")
                        logger.info(f"[Encoder] Dual encoding - Semantic grid: {semantic_grid}")
                        
                except ImportError:
                    logger.warning("[Encoder] Advanced vision processing not available, using standard VAE only")
                    ref_latent = vae.encode(image[:, :, :, :3])
                    
                if debug_mode:
                    logger.info(f"[Encoder] Encoded edit_image reference latent shape: {ref_latent.shape}")

        # Handle template application
        if use_custom_system_prompt:
            if debug_mode:
                logger.info("[Encoder] Using custom formatted text from Template Builder")
        else:
            if mode == "text_to_image":
                template = (
                    "<|im_start|>system\n"
                    "Describe the image by detailing the color, shape, size, texture, "
                    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            else: # image_edit
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
                logger.info(f"[Encoder] Applied default Qwen formatting for {mode}")

        # Tokenize
        if debug_mode:
            logger.info(f"[Encoder] Tokenizing with text: '{text[:50]}...' and {len(images_for_tokenizer)} images")

        # NOTE: The custom tokenizer logic for multi-frame has been removed, as the model
        # does not support it. The Canvas Composer node is the correct approach.
        tokens = clip.tokenize(text, images=images_for_tokenizer)

        # Handle token removal
        if token_removal == "diffsynth" and not use_custom_system_prompt:
            drop_count = 34 if mode == "text_to_image" else 64
            for key in tokens:
                for i in range(len(tokens.get(key, []))):
                    token_list = tokens[key][i]
                    if len(token_list) > drop_count:
                        tokens[key][i] = token_list[drop_count:]
            if debug_mode:
                logger.info(f"[Encoder] Dropped first {drop_count} tokens (DiffSynth style)")

        elif token_removal == "none":
            if debug_mode:
                logger.info("[Encoder] Keeping all tokens (no removal)")

        # Encode tokens using ComfyUI's method
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        # Process context_image separately (ControlNet-style)
        if context_image is not None and vae is not None:
            if debug_mode:
                logger.info(f"[Encoder] Processing context image: {context_image.shape}")

            # Use context image as-is (no resizing needed)
            context_latent = vae.encode(context_image[:, :, :, :3])
            if debug_mode:
                logger.info(f"[Encoder] Encoded context_image latent shape: {context_latent.shape}")

        # Add reference and context latents to conditioning metadata
        conditioning_updates = {}
        all_ref_latents = []
        if ref_latent is not None:
            all_ref_latents.append(ref_latent)
        if context_latent is not None:
            all_ref_latents.append(context_latent)

        # DUAL ENCODING: Add semantic-reconstructive fusion data
        if dual_encoding_data is not None:
            conditioning_updates["dual_encoding"] = dual_encoding_data
            if debug_mode:
                logger.info("[Encoder] Added dual encoding data to conditioning (semantic + reconstructive)")

        if all_ref_latents:
            conditioning_updates["reference_latents"] = all_ref_latents
            
        if conditioning_updates and COMFY_AVAILABLE:
            conditioning = node_helpers.conditioning_set_values(conditioning, conditioning_updates, append=True)
            if debug_mode:
                update_keys = list(conditioning_updates.keys())
                logger.info(f"[Encoder] Added conditioning updates: {update_keys}")

        if debug_mode:
            logger.info(f"[Encoder] Final conditioning created for mode: {mode}")

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
        import comfy.samplers
        import comfy.utils

        sampler = comfy.samplers.KSampler()

        # Stage 1: Full generation at current resolution
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
            denoise=1.0
        )[0]

        # Decode to image space
        decoded = vae.decode(stage1_samples["samples"])

        # Upscale the image
        h, w = decoded.shape[1], decoded.shape[2]
        new_h = int(h * upscale_factor)
        new_w = int(w * upscale_factor)

        decoded_chw = decoded.movedim(-1, 1)
        upscaled = comfy.utils.common_upscale(
            decoded_chw, new_w, new_h, "bicubic", "disabled"
        )
        upscaled_hwc = upscaled.movedim(1, -1)

        # Encode back to latent space
        stage2_latent = vae.encode(upscaled_hwc[:, :, :, :3])

        # Stage 2: Refinement with partial denoise
        refined_samples = sampler.sample(
            model=model,
            seed=seed + 1,
            steps=max(steps // 2, 10),
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image={"samples": stage2_latent},
            denoise=denoise
        )[0]

        return (refined_samples,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenVLCLIPLoader": QwenVLCLIPLoader,
    "QwenVLTextEncoder": QwenVLTextEncoder,
    "QwenLowresFixNode": QwenLowresFixNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLCLIPLoader": "Qwen2.5-VL CLIP Loader",
    "QwenVLTextEncoder": "Qwen2.5-VL Text Encoder",
    "QwenLowresFixNode": "Qwen Lowres Fix (Two-Stage)",
}
