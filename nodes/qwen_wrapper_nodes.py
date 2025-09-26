"""
ComfyUI nodes for proper Qwen Image Edit latent handling.

These nodes bypass ComfyUI's conditioning system for edit latents,
passing them directly to the model as DiffSynth does.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import comfy.model_management as mm

logger = logging.getLogger(__name__)


class QwenImageEncodeWrapper:
    """
    Encode images to 16-channel latents for Qwen Image Edit.

    This node encodes reference images to latents that will be concatenated
    in the sequence dimension during the forward pass.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "pad_to_even": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Pad dimensions to be divisible by 2 for patch processing"
                }),
            }
        }

    RETURN_TYPES = ("QWEN_EDIT_LATENT",)
    RETURN_NAMES = ("edit_latent",)
    FUNCTION = "encode"
    CATEGORY = "QwenImage/Wrappers"
    TITLE = "Qwen Image Encode (Edit Latent)"
    DESCRIPTION = "Encode reference images to edit latents for Qwen Image Edit"

    def encode(self, vae, image, pad_to_even=True):
        """Encode image to edit latents."""

        # VAE encode the image
        latent = vae.encode(image[:,:,:,:3])

        # Check if we need to pad for even dimensions
        if pad_to_even:
            B, C, H, W = latent.shape
            H_pad = H + (H % 2)
            W_pad = W + (W % 2)

            if H_pad != H or W_pad != W:
                latent = F.pad(latent, (0, W_pad - W, 0, H_pad - H), mode='replicate')
                logger.info(f"Padded edit latent from [{B}, {C}, {H}, {W}] to [{B}, {C}, {H_pad}, {W_pad}]")

        # Wrap in dict for type safety
        edit_latent = {
            "latent": latent,
            "is_edit": True,
            "original_shape": latent.shape
        }

        return (edit_latent,)


class QwenImageCombineLatents:
    """
    Combine multiple edit latents into a list for the model.

    This node allows combining up to 4 edit latents that will be
    concatenated in the sequence dimension.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "edit_latent1": ("QWEN_EDIT_LATENT",),
                "edit_latent2": ("QWEN_EDIT_LATENT",),
                "edit_latent3": ("QWEN_EDIT_LATENT",),
                "edit_latent4": ("QWEN_EDIT_LATENT",),
            }
        }

    RETURN_TYPES = ("QWEN_EDIT_LATENTS",)
    RETURN_NAMES = ("edit_latents",)
    FUNCTION = "combine"
    CATEGORY = "QwenImage/Wrappers"
    TITLE = "Qwen Combine Edit Latents"
    DESCRIPTION = "Combine multiple edit latents for multi-image editing"

    def combine(self, edit_latent1=None, edit_latent2=None,
                edit_latent3=None, edit_latent4=None):
        """Combine edit latents into a list."""

        latents = []
        for latent in [edit_latent1, edit_latent2, edit_latent3, edit_latent4]:
            if latent is not None:
                latents.append(latent["latent"])

        if not latents:
            return (None,)

        # Return as wrapped list
        edit_latents = {
            "latents": latents,
            "count": len(latents)
        }

        logger.info(f"Combined {len(latents)} edit latents")
        return (edit_latents,)


class QwenImageModelWithEdit:
    """
    Qwen model wrapper that properly handles edit latents.

    This node wraps the model to inject edit latents directly into
    the forward pass, bypassing ComfyUI's conditioning system.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "edit_latents": ("QWEN_EDIT_LATENTS",),
                "context_latents": ("LATENT",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "wrap_model"
    CATEGORY = "QwenImage/Wrappers"
    TITLE = "Qwen Model with Edit Latents"
    DESCRIPTION = "Wrap model to handle edit latents in forward pass"

    def wrap_model(self, model, edit_latents=None, context_latents=None):
        """Wrap model with edit latent handling."""

        # Create a wrapper that intercepts the forward pass
        class EditLatentWrapper:
            def __init__(self, base_model, edit_latents, context_latents):
                self.base_model = base_model
                self.edit_latents = edit_latents
                self.context_latents = context_latents

                # Copy attributes from base model
                for attr in dir(base_model):
                    if not attr.startswith('_') and not hasattr(self, attr):
                        try:
                            setattr(self, attr, getattr(base_model, attr))
                        except:
                            pass

            def __call__(self, x, timestep, context, **kwargs):
                # Inject edit latents into context
                if self.edit_latents is not None:
                    context = context.copy() if isinstance(context, dict) else {}
                    context["edit_latents"] = self.edit_latents["latents"]
                    logger.debug(f"Injected {len(self.edit_latents['latents'])} edit latents")

                if self.context_latents is not None:
                    context = context.copy() if isinstance(context, dict) else {}
                    context["context_latents"] = self.context_latents["samples"]

                # Call base model
                return self.base_model(x, timestep, context, **kwargs)

            def get_model_object(self, *args, **kwargs):
                if hasattr(self.base_model, 'get_model_object'):
                    return self.base_model.get_model_object(*args, **kwargs)
                return self.base_model

        # Create wrapped model
        wrapped = EditLatentWrapper(model, edit_latents, context_latents)

        logger.info(f"Created model wrapper with edit latents: {edit_latents is not None}")
        return (wrapped,)


class QwenImageSamplerWithEdit:
    """
    Custom sampler that properly handles edit latents.

    This sampler ensures edit latents are passed through the model
    during the denoising process.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "edit_latents": ("QWEN_EDIT_LATENTS",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "QwenImage/Wrappers"
    TITLE = "Qwen Sampler with Edit"
    DESCRIPTION = "KSampler that properly handles edit latents"

    def sample(self, model, positive, negative, latent, steps, cfg,
               sampler_name, scheduler, denoise, seed, edit_latents=None):
        """Sample with edit latents."""

        # If we have edit latents, wrap the model
        if edit_latents is not None:
            # Create a temporary wrapper
            class TempEditWrapper:
                def __init__(self, base_model, edit_latents):
                    self.base_model = base_model
                    self.edit_latents = edit_latents

                def __getattr__(self, name):
                    return getattr(self.base_model, name)

                def __call__(self, x, timestep, cond, **kwargs):
                    # Add edit latents to kwargs
                    kwargs["edit_latents"] = self.edit_latents["latents"]
                    return self.base_model(x, timestep, cond, **kwargs)

            model = TempEditWrapper(model, edit_latents)
            logger.info(f"Wrapped model with {len(edit_latents['latents'])} edit latents for sampling")

        # Use standard KSampler
        from nodes import KSampler
        sampler = KSampler()

        return sampler.sample(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=denoise
        )


class QwenDebugLatents:
    """
    Debug node to inspect latent dimensions and properties.

    Useful for understanding dimension mismatches and debugging
    the latent flow through the pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "main_latent": ("LATENT",),
                "edit_latent": ("QWEN_EDIT_LATENT",),
                "edit_latents": ("QWEN_EDIT_LATENTS",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "debug"
    CATEGORY = "QwenImage/Debug"
    TITLE = "Qwen Debug Latents"
    DESCRIPTION = "Debug latent dimensions and properties"
    OUTPUT_NODE = True

    def debug(self, main_latent=None, edit_latent=None, edit_latents=None):
        """Debug latent information."""

        print("\n" + "="*60)
        print("QWEN LATENT DEBUG INFORMATION")
        print("="*60)

        if main_latent is not None:
            latent = main_latent["samples"]
            print(f"\nMain Latent:")
            print(f"  Shape: {latent.shape}")
            print(f"  Dtype: {latent.dtype}")
            print(f"  Device: {latent.device}")
            print(f"  Min/Max: {latent.min().item():.3f} / {latent.max().item():.3f}")

            # Check if dimensions are even
            B, C, H, W = latent.shape
            print(f"  Dimension check:")
            print(f"    Height {H}: {'✓ Even' if H % 2 == 0 else '✗ Odd (needs padding)'}")
            print(f"    Width {W}: {'✓ Even' if W % 2 == 0 else '✗ Odd (needs padding)'}")
            print(f"  Pixel dimensions: {H*8}x{W*8}")

        if edit_latent is not None:
            latent = edit_latent["latent"]
            print(f"\nEdit Latent:")
            print(f"  Shape: {latent.shape}")
            print(f"  Original shape: {edit_latent.get('original_shape', 'Unknown')}")
            print(f"  Is edit: {edit_latent.get('is_edit', False)}")

        if edit_latents is not None:
            print(f"\nEdit Latents Collection:")
            print(f"  Count: {edit_latents['count']}")
            for i, latent in enumerate(edit_latents["latents"]):
                print(f"  Latent {i+1}: {latent.shape}")

        print("="*60 + "\n")

        return {}


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenImageEncodeWrapper": QwenImageEncodeWrapper,
    "QwenImageCombineLatents": QwenImageCombineLatents,
    "QwenImageModelWithEdit": QwenImageModelWithEdit,
    "QwenImageSamplerWithEdit": QwenImageSamplerWithEdit,
    "QwenDebugLatents": QwenDebugLatents,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEncodeWrapper": "Qwen Image Encode (Edit Latent)",
    "QwenImageCombineLatents": "Qwen Combine Edit Latents",
    "QwenImageModelWithEdit": "Qwen Model with Edit Latents",
    "QwenImageSamplerWithEdit": "Qwen Sampler with Edit",
    "QwenDebugLatents": "Qwen Debug Latents",
}