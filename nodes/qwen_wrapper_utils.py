"""
Utility functions for Qwen wrapper nodes.

Handles tensor dimension conversions, 2x2 packing/unpacking,
and other operations needed for DiffSynth compatibility.
"""

import torch
import logging
from typing import Tuple, Optional, List
from einops import rearrange

logger = logging.getLogger(__name__)

# Set up verbose logging
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def pack_2x2(tensor: torch.Tensor, P: int = 2, Q: int = 2) -> torch.Tensor:
    """
    Pack tensor using 2x2 spatial packing (following DiffSynth pattern).

    Args:
        tensor: Input tensor [B, C, H, W] or [B, C, T, H, W]
        P: Patch height (default 2)
        Q: Patch width (default 2)

    Returns:
        Packed tensor with spatial dimensions reduced by P*Q
    """
    logger.debug(f"Packing tensor: shape={tensor.shape}, P={P}, Q={Q}")

    if tensor.dim() == 4:
        # Standard 4D tensor [B, C, H, W]
        B, C, H, W = tensor.shape

        # Ensure dimensions are divisible
        if H % P != 0:
            logger.warning(f"Height {H} not divisible by {P}, padding may be needed")
        if W % Q != 0:
            logger.warning(f"Width {W} not divisible by {Q}, padding may be needed")

        # Pack using einops rearrange
        packed = rearrange(
            tensor,
            'b c (h p) (w q) -> b (c p q) h w',
            p=P, q=Q
        )
        logger.debug(f"Packed 4D: {tensor.shape} -> {packed.shape}")

    elif tensor.dim() == 5:
        # Video/multi-frame tensor [B, C, T, H, W]
        B, C, T, H, W = tensor.shape

        # Pack spatial dimensions
        packed = rearrange(
            tensor,
            'b c t (h p) (w q) -> b (c p q) t h w',
            p=P, q=Q
        )
        logger.debug(f"Packed 5D: {tensor.shape} -> {packed.shape}")

    else:
        logger.error(f"Unsupported tensor dimension: {tensor.dim()}")
        raise ValueError(f"Expected 4D or 5D tensor, got {tensor.dim()}D")

    return packed


def unpack_2x2(tensor: torch.Tensor, original_channels: int, P: int = 2, Q: int = 2) -> torch.Tensor:
    """
    Unpack tensor from 2x2 spatial packing.

    Args:
        tensor: Packed tensor
        original_channels: Number of channels before packing
        P: Patch height (default 2)
        Q: Patch width (default 2)

    Returns:
        Unpacked tensor with spatial dimensions restored
    """
    logger.debug(f"Unpacking tensor: shape={tensor.shape}, original_channels={original_channels}")

    if tensor.dim() == 4:
        # Standard 4D tensor [B, C_packed, H_packed, W_packed]
        B, C_packed, H_packed, W_packed = tensor.shape

        # Unpack using einops rearrange
        unpacked = rearrange(
            tensor,
            'b (c p q) h w -> b c (h p) (w q)',
            c=original_channels, p=P, q=Q
        )
        logger.debug(f"Unpacked 4D: {tensor.shape} -> {unpacked.shape}")

    elif tensor.dim() == 5:
        # Video/multi-frame tensor [B, C_packed, T, H_packed, W_packed]
        B, C_packed, T, H_packed, W_packed = tensor.shape

        # Unpack spatial dimensions
        unpacked = rearrange(
            tensor,
            'b (c p q) t h w -> b c t (h p) (w q)',
            c=original_channels, p=P, q=Q
        )
        logger.debug(f"Unpacked 5D: {tensor.shape} -> {unpacked.shape}")

    else:
        logger.error(f"Unsupported tensor dimension: {tensor.dim()}")
        raise ValueError(f"Expected 4D or 5D tensor, got {tensor.dim()}D")

    return unpacked


def convert_4d_to_5d(tensor: torch.Tensor, temporal_dim: int = 1) -> torch.Tensor:
    """
    Convert 4D tensor to 5D by adding temporal dimension.

    Args:
        tensor: 4D tensor [B, C, H, W]
        temporal_dim: Size of temporal dimension to add (default 1)

    Returns:
        5D tensor [B, C, T, H, W]
    """
    if tensor.dim() != 4:
        logger.warning(f"Expected 4D tensor, got {tensor.dim()}D")
        return tensor

    B, C, H, W = tensor.shape
    # Add temporal dimension
    tensor_5d = tensor.unsqueeze(2)  # [B, C, 1, H, W]

    if temporal_dim > 1:
        # Repeat along temporal dimension if needed
        tensor_5d = tensor_5d.repeat(1, 1, temporal_dim, 1, 1)

    logger.debug(f"Converted 4D to 5D: {tensor.shape} -> {tensor_5d.shape}")
    return tensor_5d


def convert_5d_to_4d(tensor: torch.Tensor, mode: str = "squeeze") -> torch.Tensor:
    """
    Convert 5D tensor to 4D by handling temporal dimension.

    Args:
        tensor: 5D tensor [B, C, T, H, W]
        mode: How to handle temporal dimension
            - "squeeze": Remove if T=1
            - "first": Take first frame
            - "last": Take last frame
            - "mean": Average across frames
            - "flatten": Reshape to [B*T, C, H, W]

    Returns:
        4D tensor [B, C, H, W] or [B*T, C, H, W]
    """
    if tensor.dim() != 5:
        logger.warning(f"Expected 5D tensor, got {tensor.dim()}D")
        return tensor

    B, C, T, H, W = tensor.shape

    if mode == "squeeze":
        if T == 1:
            tensor_4d = tensor.squeeze(2)
        else:
            logger.warning(f"Cannot squeeze temporal dim with T={T}, using first frame")
            tensor_4d = tensor[:, :, 0, :, :]
    elif mode == "first":
        tensor_4d = tensor[:, :, 0, :, :]
    elif mode == "last":
        tensor_4d = tensor[:, :, -1, :, :]
    elif mode == "mean":
        tensor_4d = tensor.mean(dim=2)
    elif mode == "flatten":
        tensor_4d = rearrange(tensor, 'b c t h w -> (b t) c h w')
    else:
        logger.error(f"Unknown mode: {mode}")
        raise ValueError(f"Unknown conversion mode: {mode}")

    logger.debug(f"Converted 5D to 4D ({mode}): {tensor.shape} -> {tensor_4d.shape}")
    return tensor_4d


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 32, mode: str = "constant") -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Pad tensor dimensions to be divisible by multiple.

    Args:
        tensor: Input tensor [B, C, H, W] or [B, C, T, H, W]
        multiple: Target multiple (default 32 for VAE)
        mode: Padding mode (constant, reflect, replicate)

    Returns:
        Padded tensor and padding amounts (left, right, top, bottom)
    """
    if tensor.dim() == 4:
        B, C, H, W = tensor.shape
    elif tensor.dim() == 5:
        B, C, T, H, W = tensor.shape
    else:
        logger.error(f"Unsupported tensor dimension: {tensor.dim()}")
        raise ValueError(f"Expected 4D or 5D tensor")

    # Calculate padding needed
    h_pad = (multiple - H % multiple) % multiple
    w_pad = (multiple - W % multiple) % multiple

    # Distribute padding evenly (prefer right/bottom if odd)
    top = h_pad // 2
    bottom = h_pad - top
    left = w_pad // 2
    right = w_pad - left

    padding = (left, right, top, bottom)

    if h_pad > 0 or w_pad > 0:
        logger.debug(f"Padding tensor from H={H},W={W} to H={H+h_pad},W={W+w_pad}")

        if tensor.dim() == 4:
            padded = torch.nn.functional.pad(tensor, padding, mode=mode)
        else:
            # For 5D, pad spatial dimensions only
            padded = torch.nn.functional.pad(tensor, padding + (0, 0), mode=mode)
    else:
        logger.debug("No padding needed, dimensions already aligned")
        padded = tensor

    return padded, padding


def unpad_tensor(tensor: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    Remove padding from tensor.

    Args:
        tensor: Padded tensor
        padding: Padding amounts (left, right, top, bottom)

    Returns:
        Unpadded tensor
    """
    left, right, top, bottom = padding

    if tensor.dim() == 4:
        B, C, H, W = tensor.shape
        unpadded = tensor[:, :, top:H-bottom if bottom > 0 else H, left:W-right if right > 0 else W]
    elif tensor.dim() == 5:
        B, C, T, H, W = tensor.shape
        unpadded = tensor[:, :, :, top:H-bottom if bottom > 0 else H, left:W-right if right > 0 else W]
    else:
        logger.error(f"Unsupported tensor dimension: {tensor.dim()}")
        raise ValueError(f"Expected 4D or 5D tensor")

    logger.debug(f"Unpadded tensor: {tensor.shape} -> {unpadded.shape}")
    return unpadded


def concatenate_edit_latents(main_latent: torch.Tensor, edit_latents: List[torch.Tensor]) -> torch.Tensor:
    """
    Concatenate edit latents in sequence dimension (following DiffSynth).

    Args:
        main_latent: Main generation latent [B, C, H, W]
        edit_latents: List of edit reference latents

    Returns:
        Concatenated tensor with edit latents in sequence dimension
    """
    logger.debug(f"Concatenating {len(edit_latents)} edit latents to main latent")

    # Ensure all tensors are 5D
    if main_latent.dim() == 4:
        main_latent = convert_4d_to_5d(main_latent)

    tensors_to_concat = [main_latent]

    for i, edit_latent in enumerate(edit_latents):
        if edit_latent.dim() == 4:
            edit_latent = convert_4d_to_5d(edit_latent)

        # Validate dimensions match
        if edit_latent.shape[1:] != main_latent.shape[1:]:
            logger.warning(f"Edit latent {i} shape mismatch: {edit_latent.shape} vs {main_latent.shape}")
            # Could add resizing logic here if needed

        tensors_to_concat.append(edit_latent)

    # Concatenate along sequence dimension (dim=2)
    concatenated = torch.cat(tensors_to_concat, dim=2)
    logger.info(f"Concatenated latents: {main_latent.shape} + {len(edit_latents)} edits -> {concatenated.shape}")

    return concatenated


# Utility functions for debugging
def print_tensor_stats(tensor: torch.Tensor, name: str = "Tensor"):
    """Print detailed tensor statistics for debugging."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Statistics:")
    logger.info(f"  Shape: {tensor.shape}")
    logger.info(f"  Dtype: {tensor.dtype}")
    logger.info(f"  Device: {tensor.device}")
    logger.info(f"  Min: {tensor.min().item():.4f}")
    logger.info(f"  Max: {tensor.max().item():.4f}")
    logger.info(f"  Mean: {tensor.mean().item():.4f}")
    logger.info(f"  Std: {tensor.std().item():.4f}")
    logger.info(f"  NaN count: {torch.isnan(tensor).sum().item()}")
    logger.info(f"  Inf count: {torch.isinf(tensor).sum().item()}")
    logger.info(f"{'='*60}\n")