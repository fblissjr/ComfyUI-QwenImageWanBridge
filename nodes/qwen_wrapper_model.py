"""
Qwen Image Model Wrapper implementing DiffSynth's model_fn_qwen_image.

This module wraps the Qwen Image DiT model to properly handle the forward pass
with all the critical operations from DiffSynth, including:
- The packing operation (rearrange with P=2, Q=2)
- Edit latents concatenation in sequence dimension
- Proper handling of context latents
- Entity control support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
import comfy.model_management as mm
import comfy.model_base
from .qwen_wrapper_base import QwenWrapperBase, PATCH_SIZE
from .qwen_wrapper_utils import (
    pack_2x2, unpack_2x2, convert_4d_to_5d, convert_5d_to_4d,
    pad_to_multiple, unpad_tensor, concatenate_edit_latents,
    print_tensor_stats
)

logger = logging.getLogger(__name__)


class QwenImageModelWrapper(nn.Module, QwenWrapperBase):
    """
    Wrapper for Qwen Image DiT model following DiffSynth's model_fn_qwen_image.

    This wrapper implements the critical forward pass logic from DiffSynth,
    ensuring proper handling of the 16-channel VAE latents and the 2x2 patch
    packing operation that's essential for Qwen's architecture.
    """

    def __init__(self, dit_model=None, model_config=None):
        super().__init__()
        self.dit = dit_model
        self.model_config = model_config or {}
        self.device = mm.get_torch_device()
        self.offload_device = mm.unet_offload_device()

        # Model parameters from config
        self.in_channels = self.model_config.get("in_channels", 16)
        self.out_channels = self.model_config.get("out_channels", 16)
        self.hidden_size = self.model_config.get("hidden_size", 3584)  # Qwen2.5-VL 7B

    def forward(self, x: torch.Tensor, timestep: torch.Tensor,
                context: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Forward pass implementing model_fn_qwen_image from DiffSynth.

        Args:
            x: Input latents [B, C, H, W] with C=16 for Qwen
            timestep: Timestep tensor
            context: Dictionary containing conditioning information

        Returns:
            Output latents [B, C, H, W]
        """
        return self.model_fn_qwen_image(
            latents=x,
            timestep=timestep,
            **context,
            **kwargs
        )

    def model_fn_qwen_image(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor = None,
        prompt_emb_mask: torch.Tensor = None,
        height: int = None,
        width: int = None,
        edit_latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        context_latents: Optional[torch.Tensor] = None,
        entity_prompt_emb: Optional[torch.Tensor] = None,
        entity_prompt_emb_mask: Optional[torch.Tensor] = None,
        entity_masks: Optional[torch.Tensor] = None,
        edit_rope_interpolation: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Implementation of model_fn_qwen_image from DiffSynth line 799.

        This is the critical function that properly formats inputs for Qwen DiT.
        """
        if self.dit is None:
            # Fallback for testing
            return torch.randn_like(latents)

        # Get dimensions from latents if not provided
        if height is None or width is None:
            # Latents are at 1/8 scale, patches reduce by 2x more
            height = latents.shape[2] * 8
            width = latents.shape[3] * 8

        # Handle odd dimensions by padding the latents first
        B, C, H_latent, W_latent = latents.shape
        H_pad = H_latent + (H_latent % 2)  # Make even if odd
        W_pad = W_latent + (W_latent % 2)  # Make even if odd

        if H_pad != H_latent or W_pad != W_latent:
            latents = F.pad(latents, (0, W_pad - W_latent, 0, H_pad - H_latent), mode='replicate')
            logger.debug(f"Padded main latents from [{B}, {C}, {H_latent}, {W_latent}] to [{B}, {C}, {H_pad}, {W_pad}]")

        # Build image shapes list for position embeddings using padded dimensions
        img_shapes = [(latents.shape[0], latents.shape[2]//2, latents.shape[3]//2)]

        # Get text sequence lengths from mask
        if prompt_emb_mask is not None:
            txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        else:
            txt_seq_lens = [prompt_emb.shape[1]] * prompt_emb.shape[0]

        # Convert timestep to DiffSynth format (divide by 1000)
        if timestep.max() > 1:
            timestep = timestep / 1000

        # CRITICAL: Apply packing operation using utility
        # This transforms [B, C, H, W] -> [B, (H/2 * W/2), (C * 2 * 2)]
        # Debug logging for input
        if logger.isEnabledFor(logging.DEBUG):
            print_tensor_stats(latents, "Input Latents")

        # Use utility for 2x2 packing then rearrange to sequence format
        image_packed = pack_2x2(latents, P=2, Q=2)  # [B, C*4, H/2, W/2]
        B, C_packed, H_packed, W_packed = image_packed.shape

        # Rearrange to sequence format for transformer
        image = rearrange(
            image_packed,
            'b c h w -> b (h w) c'
        )
        image_seq_len = image.shape[1]

        # Handle context latents (for ControlNet-style conditioning)
        if context_latents is not None:
            # Pad context latents if needed
            B_ctx, C_ctx, H_ctx, W_ctx = context_latents.shape
            H_ctx_pad = H_ctx + (H_ctx % 2)
            W_ctx_pad = W_ctx + (W_ctx % 2)

            if H_ctx_pad != H_ctx or W_ctx_pad != W_ctx:
                context_latents = F.pad(context_latents, (0, W_ctx_pad - W_ctx, 0, H_ctx_pad - H_ctx), mode='replicate')

            img_shapes.append((context_latents.shape[0],
                              context_latents.shape[2]//2,
                              context_latents.shape[3]//2))
            context_image = rearrange(
                context_latents,
                "B C (H P) (W Q) -> B (H W) (C P Q)",
                H=context_latents.shape[2]//2,
                W=context_latents.shape[3]//2,
                P=2, Q=2
            )
            image = torch.cat([image, context_image], dim=1)

        # Handle edit latents (for image editing)
        if edit_latents is not None:
            edit_latents_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
            edit_images = []
            for e in edit_latents_list:
                # Pad edit latents if needed
                B_e, C_e, H_e, W_e = e.shape
                H_e_pad = H_e + (H_e % 2)
                W_e_pad = W_e + (W_e % 2)

                if H_e_pad != H_e or W_e_pad != W_e:
                    e = F.pad(e, (0, W_e_pad - W_e, 0, H_e_pad - H_e), mode='replicate')

                img_shapes.append((e.shape[0], e.shape[2]//2, e.shape[3]//2))
                edit_image = rearrange(
                    e,
                    "B C (H P) (W Q) -> B (H W) (C P Q)",
                    H=e.shape[2]//2, W=e.shape[3]//2,
                    P=2, Q=2
                )
                edit_images.append(edit_image)
            # CRITICAL: Concatenate in sequence dimension (dim=1), not batch or channel!
            image = torch.cat([image] + edit_images, dim=1)

        # Apply input projection if available
        if hasattr(self.dit, 'img_in'):
            image = self.dit.img_in(image)

        # Get time embedding
        if hasattr(self.dit, 'time_text_embed'):
            conditioning = self.dit.time_text_embed(timestep, image.dtype)
        else:
            # Fallback time embedding
            conditioning = self.get_time_embedding(timestep, image.dtype)

        # Process text embeddings
        if prompt_emb is not None:
            if hasattr(self.dit, 'txt_in') and hasattr(self.dit, 'txt_norm'):
                text = self.dit.txt_in(self.dit.txt_norm(prompt_emb))
            else:
                # Direct text embedding
                text = prompt_emb

            # Apply RoPE if needed
            if hasattr(self.dit, 'compute_rope') and edit_rope_interpolation:
                image_rotary_emb = self.dit.compute_rope(img_shapes, edit_rope_interpolation)
            else:
                image_rotary_emb = None
        else:
            text = None
            image_rotary_emb = None

        # Handle entity control if provided
        if entity_prompt_emb is not None and hasattr(self.dit, 'process_entity_masks'):
            text, image_rotary_emb, attention_mask = self.dit.process_entity_masks(
                latents, prompt_emb, prompt_emb_mask,
                entity_prompt_emb, entity_prompt_emb_mask,
                entity_masks, height, width, image, img_shapes
            )

        # Build attention mask
        if prompt_emb_mask is not None:
            attention_mask = prompt_emb_mask
        else:
            attention_mask = torch.ones((prompt_emb.shape[0], prompt_emb.shape[1]),
                                      device=prompt_emb.device, dtype=torch.bool)

        # Call the DiT forward pass
        if hasattr(self.dit, 'forward_features'):
            # DiffSynth-style forward
            output = self.dit.forward_features(
                image=image,
                text=text,
                timestep=conditioning,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens
            )
        elif hasattr(self.dit, 'forward'):
            # Standard forward with available arguments
            forward_kwargs = {
                'x': image,
                'timestep': conditioning
            }
            if text is not None:
                forward_kwargs['context'] = text
            if attention_mask is not None:
                forward_kwargs['attention_mask'] = attention_mask

            output = self.dit(**forward_kwargs)
        else:
            # Fallback
            output = image

        # Extract the main generation part (first image_seq_len tokens)
        if output.shape[1] > image_seq_len:
            output = output[:, :image_seq_len]

        # Apply output projection if available
        if hasattr(self.dit, 'img_out'):
            output = self.dit.img_out(output)

        # CRITICAL: Unpack from sequence back to spatial format
        # Reverse of line 802: [B, seq, features] -> [B, C, H, W]
        # Use the same patch counts as in packing
        output = rearrange(
            output,
            "B (H W) (C P Q) -> B C (H P) (W Q)",
            H=H_patches, W=W_patches, P=2, Q=2
        )

        # Remove padding if we added any
        if H_pad != H_latent or W_pad != W_latent:
            output = output[:, :, :H_latent, :W_latent]

        return output

    def get_time_embedding(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Fallback time embedding if model doesn't have time_text_embed."""
        # Simple sinusoidal embedding
        half_dim = self.hidden_size // 2
        emb = torch.exp(
            -torch.arange(half_dim, dtype=dtype, device=timestep.device)
            * (torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def process_edit_latents(self, edit_latents: Union[torch.Tensor, List[torch.Tensor]],
                           main_latents: torch.Tensor) -> torch.Tensor:
        """
        Process edit latents for concatenation.

        This is handled inside model_fn_qwen_image but exposed for debugging.
        """
        return self.handle_edit_latents(edit_latents, main_latents)

    def apply_packing(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Apply the 2x2 spatial packing operation.

        Exposed for testing and debugging.
        """
        return self.pack_latents(latents, height, width)

    def get_model_options(self) -> Dict[str, Any]:
        """Get model options for ComfyUI compatibility."""
        return {
            "transformer_options": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "hidden_size": self.hidden_size,
                "patch_size": PATCH_SIZE
            }
        }


class QwenImageModelWrapperNode:
    """
    ComfyUI node for the Qwen Image Model Wrapper.

    This node wraps the loaded DiT model with proper forward pass logic.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN_IMAGE_MODEL",),
            },
            "optional": {
                "enable_edit_rope": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable RoPE interpolation for edit mode"
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable debug logging"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "wrap_model"
    CATEGORY = "QwenImage/Wrappers"
    TITLE = "Qwen Image Model Wrapper"
    DESCRIPTION = "Wrap Qwen DiT with DiffSynth forward pass logic"

    def wrap_model(self, model, enable_edit_rope=False, debug_mode=False):
        """Wrap the model with proper forward pass."""

        # Extract the actual DiT from the loaded model
        if hasattr(model, 'diffusion_model'):
            dit = model.diffusion_model
        elif hasattr(model, 'model'):
            dit = model.model
        else:
            dit = model

        # Get model config
        if hasattr(model, 'model_config'):
            config = model.model_config
        else:
            config = {}

        # Create wrapper
        wrapper = QwenImageModelWrapper(dit, config)
        wrapper.edit_rope_interpolation = enable_edit_rope

        if debug_mode:
            logger.info(f"Created Qwen model wrapper with config: {config}")
            logger.info(f"Model has methods: {[m for m in dir(dit) if not m.startswith('_')][:10]}")

        # Wrap in ComfyUI model base for compatibility
        class WrappedQwenModel(comfy.model_base.BaseModel):
            def __init__(self, wrapper, config):
                super().__init__(config)
                self.diffusion_model = wrapper

            def get_model_object(self, *args, **kwargs):
                return self.diffusion_model

        # Create ComfyUI-compatible model
        comfy_model = WrappedQwenModel(wrapper, model.model_config if hasattr(model, 'model_config') else comfy.model_base.ModelConfig())

        return (comfy_model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenImageModelWrapperNode": QwenImageModelWrapperNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageModelWrapperNode": "Qwen Image Model Wrapper",
}