"""
Qwen Image Model Wrapper V2 - DiffSynth-aligned implementation.

This module wraps the model_fn_qwen_image logic from DiffSynth into
ComfyUI-compatible nodes, preserving all critical operations.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from .qwen_wrapper_base import (
    QwenWrapperBase,
    QwenConditioningWrapper,
    QwenVAEWrapper,
    QWEN_VAE_CHANNELS,
    PATCH_SIZE
)

logger = logging.getLogger(__name__)


class QwenImageModelFn:
    """
    Direct implementation of model_fn_qwen_image from DiffSynth.

    This class encapsulates the exact logic from DiffSynth's pipeline,
    ensuring proper handling of all the model inputs and transformations.
    """

    def __init__(self, dit_model=None, device="cuda", dtype=torch.float16):
        self.dit = dit_model
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_emb_mask: torch.Tensor,
        height: int,
        width: int,
        edit_latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        context_latents: Optional[torch.Tensor] = None,
        entity_prompt_emb: Optional[torch.Tensor] = None,
        entity_prompt_emb_mask: Optional[torch.Tensor] = None,
        entity_masks: Optional[torch.Tensor] = None,
        blockwise_controlnet: Optional[Any] = None,
        blockwise_controlnet_conditioning: Optional[List] = None,
        blockwise_controlnet_inputs: Optional[List] = None,
        progress_id: int = 0,
        num_inference_steps: int = 30,
        enable_fp8_attention: bool = False,
        use_gradient_checkpointing: bool = False,
        edit_rope_interpolation: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Execute the model forward pass following DiffSynth's exact logic.

        This is a direct translation of model_fn_qwen_image from
        diffsynth/pipelines/qwen_image.py lines 774-861.
        """
        # Line 798: Calculate image shapes
        img_shapes = [(latents.shape[0], latents.shape[2]//2, latents.shape[3]//2)]

        # Line 799: Get text sequence lengths
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()

        # Line 800: Normalize timestep to [0, 1]
        timestep = timestep / 1000

        # Line 802: CRITICAL packing operation
        # Transform from [B, C, H, W] to [B, (H*W), (C*4)]
        image = rearrange(
            latents,
            "B C (H P) (W Q) -> B (H W) (C P Q)",
            H=height//16,
            W=width//16,
            P=PATCH_SIZE,
            Q=PATCH_SIZE
        )
        image_seq_len = image.shape[1]

        # Lines 805-808: Handle context latents
        if context_latents is not None:
            img_shapes.append(
                (context_latents.shape[0],
                 context_latents.shape[2]//2,
                 context_latents.shape[3]//2)
            )
            context_image = rearrange(
                context_latents,
                "B C (H P) (W Q) -> B (H W) (C P Q)",
                H=context_latents.shape[2]//2,
                W=context_latents.shape[3]//2,
                P=PATCH_SIZE,
                Q=PATCH_SIZE
            )
            image = torch.cat([image, context_image], dim=1)

        # Lines 809-813: Handle edit latents
        # CRITICAL: These are CONCATENATED, not added!
        if edit_latents is not None:
            edit_latents_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
            img_shapes.extend([
                (e.shape[0], e.shape[2]//2, e.shape[3]//2)
                for e in edit_latents_list
            ])
            edit_images = [
                rearrange(
                    e,
                    "B C (H P) (W Q) -> B (H W) (C P Q)",
                    H=e.shape[2]//2,
                    W=e.shape[3]//2,
                    P=PATCH_SIZE,
                    Q=PATCH_SIZE
                )
                for e in edit_latents_list
            ]
            image = torch.cat([image] + edit_images, dim=1)

        # If we don't have the actual DiT model, we need to simulate its operations
        if self.dit is None:
            # Fallback implementation
            return self._fallback_forward(
                image, timestep, prompt_emb, prompt_emb_mask,
                height, width, image_seq_len, img_shapes, txt_seq_lens,
                edit_rope_interpolation=edit_rope_interpolation
            )

        # Line 815: Image input projection
        image = self.dit.img_in(image)

        # Line 816: Time and text embedding
        conditioning = self.dit.time_text_embed(timestep, image.dtype)

        # Lines 818-829: Handle entity control or standard processing
        if entity_prompt_emb is not None:
            text, image_rotary_emb, attention_mask = self.dit.process_entity_masks(
                latents, prompt_emb, prompt_emb_mask,
                entity_prompt_emb, entity_prompt_emb_mask,
                entity_masks, height, width, image, img_shapes
            )
        else:
            # Line 824: Text input projection
            text = self.dit.txt_in(self.dit.txt_norm(prompt_emb))

            # Lines 825-828: Position embeddings with optional interpolation
            if edit_rope_interpolation and hasattr(self.dit.pos_embed, 'forward_sampling'):
                image_rotary_emb = self.dit.pos_embed.forward_sampling(
                    img_shapes, txt_seq_lens, device=latents.device
                )
            else:
                image_rotary_emb = self.dit.pos_embed(
                    img_shapes, txt_seq_lens, device=latents.device
                )
            attention_mask = None

        # Lines 831-833: ControlNet preprocessing
        if blockwise_controlnet is not None and blockwise_controlnet_conditioning is not None:
            blockwise_controlnet_conditioning = blockwise_controlnet.preprocess(
                blockwise_controlnet_inputs, blockwise_controlnet_conditioning
            )

        # Lines 835-854: Transformer blocks
        for block_id, block in enumerate(self.dit.transformer_blocks):
            # Process through transformer block
            if use_gradient_checkpointing:
                # With gradient checkpointing
                def block_fn(image, text):
                    return block(
                        image=image,
                        text=text,
                        temb=conditioning,
                        image_rotary_emb=image_rotary_emb,
                        attention_mask=attention_mask,
                        enable_fp8_attention=enable_fp8_attention
                    )
                text, image = torch.utils.checkpoint.checkpoint(block_fn, image, text)
            else:
                # Normal forward
                text, image = block(
                    image=image,
                    text=text,
                    temb=conditioning,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=attention_mask,
                    enable_fp8_attention=enable_fp8_attention
                )

            # Lines 847-854: Apply ControlNet if present
            if blockwise_controlnet is not None and blockwise_controlnet_conditioning is not None:
                image_slice = image[:, :image_seq_len].clone()
                controlnet_output = blockwise_controlnet.blockwise_forward(
                    image=image_slice,
                    conditionings=blockwise_controlnet_conditioning,
                    controlnet_inputs=blockwise_controlnet_inputs,
                    block_id=block_id,
                    progress_id=progress_id,
                    num_inference_steps=num_inference_steps
                )
                image[:, :image_seq_len] = image_slice + controlnet_output

        # Lines 856-858: Output normalization and projection
        image = self.dit.norm_out(image, conditioning)
        image = self.dit.proj_out(image)
        image = image[:, :image_seq_len]  # Take only the main image sequence

        # Line 860: Unpack back to latent space
        latents = rearrange(
            image,
            "B (H W) (C P Q) -> B C (H P) (W Q)",
            H=height//16,
            W=width//16,
            P=PATCH_SIZE,
            Q=PATCH_SIZE
        )

        return latents

    def _fallback_forward(
        self,
        image: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_emb_mask: torch.Tensor,
        height: int,
        width: int,
        image_seq_len: int,
        img_shapes: List[Tuple],
        txt_seq_lens: List[int],
        edit_rope_interpolation: bool = False
    ) -> torch.Tensor:
        """
        Fallback implementation when DiT model is not available.

        This provides a minimal implementation for testing the wrapper logic.
        """
        logger.warning("Using fallback model_fn implementation - no DiT model available")

        # Simulate some processing
        # In reality, this would be the full transformer computation
        B, seq_len, hidden_dim = image.shape

        # Apply some basic transformations to simulate model behavior
        # Add text conditioning influence
        if prompt_emb is not None:
            # Simulate cross-attention influence
            text_influence = prompt_emb.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
            text_influence = text_influence.expand(-1, seq_len, -1)
            image = image + 0.1 * text_influence

        # Apply timestep conditioning
        time_scale = 1.0 - timestep.item()  # Convert to denoising scale
        image = image * time_scale

        # Slice back to main sequence
        image = image[:, :image_seq_len]

        # Unpack to latent format
        latents = rearrange(
            image,
            "B (H W) (C P Q) -> B C (H P) (W Q)",
            H=height//16,
            W=width//16,
            P=PATCH_SIZE,
            Q=PATCH_SIZE
        )

        return latents


class QwenImageModelWrapper(QwenWrapperBase):
    """
    ComfyUI node wrapper for Qwen Image model with DiffSynth logic.

    This node serves as a bridge between ComfyUI's system and DiffSynth's
    proven model implementation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "cfg_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "edit_mode": ("BOOLEAN", {"default": False}),
                "edit_rope_interpolation": ("BOOLEAN", {"default": False}),
                "enable_fp8": ("BOOLEAN", {"default": False}),
                "use_gradient_checkpointing": ("BOOLEAN", {"default": False}),
                "height": ("INT", {
                    "default": 1328,
                    "min": 256,
                    "max": 4096,
                    "step": 32
                }),
                "width": ("INT", {
                    "default": 1328,
                    "min": 256,
                    "max": 4096,
                    "step": 32
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "wrap_model"
    CATEGORY = "QwenWrapper"
    DISPLAY_NAME = "Qwen Model Wrapper (DiffSynth)"

    def wrap_model(
        self,
        model,
        positive,
        negative,
        latent,
        cfg_scale=4.0,
        edit_mode=False,
        edit_rope_interpolation=False,
        enable_fp8=False,
        use_gradient_checkpointing=False,
        height=1328,
        width=1328
    ):
        """
        Wrap the model to use DiffSynth's model_fn logic.
        """
        # Ensure resolution is compatible
        height, width = self.prepare_resolution(height, width)

        # Extract latent samples
        if isinstance(latent, dict):
            latent_samples = latent.get("samples")
        else:
            latent_samples = latent

        # Ensure 16-channel format
        latent_samples = self.ensure_16_channel(latent_samples)

        # Create model function wrapper
        model_fn = QwenImageModelFn(
            dit_model=getattr(model, 'dit', None),
            device=latent_samples.device,
            dtype=latent_samples.dtype
        )

        # Extract conditioning components
        pos_cond = QwenConditioningWrapper.unpack_conditioning(positive)
        neg_cond = QwenConditioningWrapper.unpack_conditioning(negative)

        # Create a wrapped model that intercepts the forward pass
        class WrappedModel:
            def __init__(self, base_model, model_fn_instance, height, width, cfg_scale):
                self.base_model = base_model
                self.model_fn = model_fn_instance
                self.height = height
                self.width = width
                self.cfg_scale = cfg_scale
                self.pos_conditioning = pos_cond
                self.neg_conditioning = neg_cond
                self.edit_rope_interpolation = edit_rope_interpolation
                self.enable_fp8 = enable_fp8
                self.use_gradient_checkpointing = use_gradient_checkpointing

                # Copy necessary attributes from base model
                for attr in ['model_sampling', 'model_config', 'load_device',
                           'offload_device', 'model_dtype', 'manual_cast_dtype']:
                    if hasattr(base_model, attr):
                        setattr(self, attr, getattr(base_model, attr))

            def apply_model(self, x, t, c_concat=None, c_crossattn=None,
                          control=None, transformer_options=None, **kwargs):
                """
                Override the model's apply_model to use DiffSynth logic.
                """
                # Use our model_fn with proper inputs
                timestep = t

                # Get positive noise prediction
                noise_pred_pos = self.model_fn.forward(
                    latents=x,
                    timestep=timestep,
                    prompt_emb=self.pos_conditioning.get("prompt_emb"),
                    prompt_emb_mask=self.pos_conditioning.get("prompt_emb_mask"),
                    height=self.height,
                    width=self.width,
                    edit_latents=self.pos_conditioning.get("edit_latents"),
                    context_latents=self.pos_conditioning.get("context_latents"),
                    edit_rope_interpolation=self.edit_rope_interpolation,
                    enable_fp8_attention=self.enable_fp8,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                    **kwargs
                )

                # Apply CFG if needed
                if self.cfg_scale != 1.0 and self.neg_conditioning:
                    noise_pred_neg = self.model_fn.forward(
                        latents=x,
                        timestep=timestep,
                        prompt_emb=self.neg_conditioning.get("prompt_emb"),
                        prompt_emb_mask=self.neg_conditioning.get("prompt_emb_mask"),
                        height=self.height,
                        width=self.width,
                        edit_latents=self.neg_conditioning.get("edit_latents"),
                        context_latents=self.neg_conditioning.get("context_latents"),
                        edit_rope_interpolation=self.edit_rope_interpolation,
                        enable_fp8_attention=self.enable_fp8,
                        use_gradient_checkpointing=self.use_gradient_checkpointing,
                        **kwargs
                    )

                    # CFG formula: neg + scale * (pos - neg)
                    noise_pred = noise_pred_neg + self.cfg_scale * (noise_pred_pos - noise_pred_neg)
                else:
                    noise_pred = noise_pred_pos

                return noise_pred

            def __getattr__(self, name):
                # Fallback to base model for other attributes
                return getattr(self.base_model, name)

        # Create and return wrapped model
        wrapped = WrappedModel(
            model,
            model_fn,
            height,
            width,
            cfg_scale
        )

        logger.info(f"Created DiffSynth-wrapped model for {height}x{width} generation")
        return (wrapped,)


# Additional node for handling the packing operation separately
class QwenLatentPackerNode(QwenWrapperBase):
    """
    Node to explicitly pack/unpack latents using DiffSynth's transformation.

    This can be useful for debugging or when you need explicit control
    over the packing operation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "operation": (["pack", "unpack"],),
                "height": ("INT", {
                    "default": 1328,
                    "min": 256,
                    "max": 4096,
                    "step": 32
                }),
                "width": ("INT", {
                    "default": 1328,
                    "min": 256,
                    "max": 4096,
                    "step": 32
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process_latent"
    CATEGORY = "QwenWrapper/Utils"
    DISPLAY_NAME = "Qwen Latent Packer (DiffSynth)"

    def process_latent(self, latent, operation, height, width):
        """
        Pack or unpack latents using DiffSynth's transformation.
        """
        # Extract samples
        if isinstance(latent, dict):
            samples = latent.get("samples")
        else:
            samples = latent

        # Ensure 16-channel format
        samples = self.ensure_16_channel(samples)

        # Ensure compatible dimensions
        height, width = self.prepare_resolution(height, width)

        if operation == "pack":
            # Apply packing transformation
            packed = self.pack_latents(samples, height, width)
            result = {"samples": packed}
        else:
            # Apply unpacking transformation
            unpacked = self.unpack_latents(samples, height, width)
            result = {"samples": unpacked}

        logger.info(f"{operation.capitalize()}ed latents: {samples.shape} -> {result['samples'].shape}")
        return (result,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "QwenImageModelWrapper": QwenImageModelWrapper,
    "QwenLatentPackerNode": QwenLatentPackerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageModelWrapper": "Qwen Model Wrapper (Wrapper)",
    "QwenLatentPackerNode": "Qwen Latent Packer (Wrapper)",
}
