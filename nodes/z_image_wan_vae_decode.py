"""
Experimental: Decode Z-Image latents using Wan VAE.

WARNING: This is mixing incompatible VAE architectures.
- Z-Image uses a Flux-derived VAE (4D tensors, scaling_factor=0.3611, shift_factor=0.1159)
- Wan VAE is a video VAE (5D tensors, uses *2-1/+1/2 normalization)

This node adds a temporal dimension to make shapes compatible and attempts
to correct for the different scaling factors. Results are still experimental
but may be usable for some workflows.

Use at your own risk. This exists for testing and experimentation.
"""

import torch
import comfy.model_management as model_management


# Known VAE scaling factors
FLUX_SCALING_FACTOR = 0.3611
FLUX_SHIFT_FACTOR = 0.1159
# Wan VAE uses process_input: x * 2.0 - 1.0, process_output: (x + 1.0) / 2.0
# This is equivalent to scaling_factor=0.5, shift_factor=0.5 in a different formulation


class ZImageWanVAEDecode:
    """
    Experimental: Decode Z-Image 4D latents using Wan 5D VAE.

    Attempts to correct for the scaling factor mismatch between Flux VAE
    (used by Z-Image) and Wan VAE. Exposes parameters for manual tuning.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
            "optional": {
                "apply_scaling_correction": ("BOOLEAN", {"default": True}),
                # Flux VAE parameters (source)
                "flux_scaling_factor": ("FLOAT", {
                    "default": FLUX_SCALING_FACTOR,
                    "min": 0.01, "max": 2.0, "step": 0.0001,
                    "tooltip": "Flux VAE scaling factor (default: 0.3611)"
                }),
                "flux_shift_factor": ("FLOAT", {
                    "default": FLUX_SHIFT_FACTOR,
                    "min": -1.0, "max": 1.0, "step": 0.0001,
                    "tooltip": "Flux VAE shift factor (default: 0.1159)"
                }),
                # Manual correction overrides
                "scale_correction": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1, "max": 5.0, "step": 0.01,
                    "tooltip": "Additional scale multiplier for manual tuning"
                }),
                "shift_correction": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Additional shift offset for manual tuning"
                }),
                # Output adjustments
                "brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip": "Post-decode brightness adjustment"
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip": "Post-decode contrast adjustment"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "debug_info")
    FUNCTION = "decode"
    CATEGORY = "ZImage/Experimental"
    DESCRIPTION = "EXPERIMENTAL: Decode Z-Image latents with Wan VAE. Includes scaling factor correction for the Flux->Wan VAE mismatch."

    def decode(self, samples, vae,
               apply_scaling_correction=True,
               flux_scaling_factor=FLUX_SCALING_FACTOR,
               flux_shift_factor=FLUX_SHIFT_FACTOR,
               scale_correction=1.0,
               shift_correction=0.0,
               brightness=1.0,
               contrast=1.0):

        latent = samples["samples"].clone()
        debug_lines = ["=== Z-Image Wan VAE Decode (Experimental) ==="]
        debug_lines.append(f"Input latent shape: {latent.shape}")
        debug_lines.append(f"Input latent stats: mean={latent.mean().item():.4f}, std={latent.std().item():.4f}")

        # Apply scaling correction to convert Flux latent space to Wan latent space
        if apply_scaling_correction:
            # Flux VAE encodes as: latent = (x - shift) / scale
            # So to decode properly, we need: x = latent * scale + shift
            # But Wan VAE expects its own normalization
            #
            # Attempt to reverse Flux scaling and apply Wan-compatible values:
            # 1. Reverse Flux encoding: x_intermediate = latent * flux_scale + flux_shift
            # 2. Apply Wan encoding expectation: wan_latent = x_intermediate * 2 - 1 (approx)

            # Simplified correction: scale and shift the latent distribution
            # These values are empirical starting points
            correction_scale = (1.0 / flux_scaling_factor) * 0.5  # Rough Flux->Wan scale
            correction_shift = -flux_shift_factor / flux_scaling_factor

            latent = latent * correction_scale + correction_shift
            debug_lines.append(f"Applied scaling correction: scale={correction_scale:.4f}, shift={correction_shift:.4f}")

        # Apply manual corrections
        if scale_correction != 1.0 or shift_correction != 0.0:
            latent = latent * scale_correction + shift_correction
            debug_lines.append(f"Applied manual correction: scale={scale_correction:.4f}, shift={shift_correction:.4f}")

        debug_lines.append(f"Corrected latent stats: mean={latent.mean().item():.4f}, std={latent.std().item():.4f}")

        # Z-Image outputs 4D: [B, C, H, W]
        # Wan VAE expects 5D: [B, C, F, H, W]
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)  # Add frame dim: [B, C, 1, H, W]
            debug_lines.append("Added temporal dimension: 4D -> 5D")

        # Save original memory_used_decode (expects 5D for Wan)
        original_memory_fn = vae.memory_used_decode

        try:
            # Load VAE to GPU
            memory_used = vae.memory_used_decode(latent.shape, vae.vae_dtype)
            model_management.load_models_gpu([vae.patcher], memory_required=memory_used, force_full_load=vae.disable_offload)

            # Decode directly through first_stage_model
            latent_gpu = latent.to(vae.vae_dtype).to(vae.device)
            decoded = vae.first_stage_model.decode(latent_gpu)
            decoded = vae.process_output(decoded.to(vae.output_device).float())

            debug_lines.append(f"Decoded shape: {decoded.shape}")

            # If we have a frame dimension, squeeze it
            if decoded.ndim == 5:
                decoded = decoded.squeeze(2)

            # Move channels to last: [B, C, H, W] -> [B, H, W, C]
            images = decoded.movedim(1, -1)

            # Apply brightness/contrast adjustments
            if brightness != 1.0 or contrast != 1.0:
                # Contrast around 0.5 (middle gray)
                images = (images - 0.5) * contrast + 0.5
                # Brightness as multiplier
                images = images * brightness
                debug_lines.append(f"Applied brightness={brightness:.2f}, contrast={contrast:.2f}")

            # Clamp to valid range
            images = images.clamp(0, 1)

            debug_lines.append(f"Output image shape: {images.shape}")
            debug_lines.append(f"Output stats: min={images.min().item():.4f}, max={images.max().item():.4f}")
            debug_lines.append("")
            debug_lines.append("NOTE: This is experimental. Colors may still be wrong.")
            debug_lines.append("Try adjusting scale_correction and shift_correction manually.")

            return (images, "\n".join(debug_lines))

        finally:
            vae.memory_used_decode = original_memory_fn


NODE_CLASS_MAPPINGS = {
    "ZImageWanVAEDecode": ZImageWanVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageWanVAEDecode": "Z-Image Wan VAE Decode (Experimental)",
}
