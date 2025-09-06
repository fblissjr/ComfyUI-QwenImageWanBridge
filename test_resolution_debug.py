#!/usr/bin/env python3
"""
Debug script to understand why optimize_resolution=True causes issues with empty latents
"""

import math

# Qwen resolutions from the node
QWEN_RESOLUTIONS = [
    (256, 256), (256, 512), (256, 768), (256, 1024), (256, 1280), (256, 1536), (256, 1792),
    (512, 256), (512, 512), (512, 768), (512, 1024), (512, 1280), (512, 1536), (512, 1792),
    (768, 256), (768, 512), (768, 768), (768, 1024), (768, 1280), (768, 1536),
    (1024, 256), (1024, 512), (1024, 768), (1024, 1024), (1024, 1280), (1024, 1536),
    (1280, 256), (1280, 512), (1280, 768), (1280, 1024), (1280, 1280),
    (1536, 256), (1536, 512), (1536, 768), (1536, 1024),
    (1792, 256), (1792, 512)
]

def get_optimal_resolution(width: int, height: int) -> tuple:
    """Find the nearest Qwen-supported resolution"""
    target_pixels = width * height
    
    # Find closest resolution by total pixels
    best_res = min(
        QWEN_RESOLUTIONS,
        key=lambda r: abs(r[0] * r[1] - target_pixels)
    )
    
    return best_res

def get_standard_resolution(width: int, height: int) -> tuple:
    """Scale to approximately 1M pixels (1024x1024)"""
    total = int(1024 * 1024)
    scale_by = math.sqrt(total / (width * height))
    new_width = round(width * scale_by)
    new_height = round(height * scale_by)
    return new_width, new_height

# Test with common image sizes
test_sizes = [
    (512, 512),    # Square small
    (1024, 1024),  # Square medium
    (2048, 2048),  # Square large
    (1920, 1080),  # 16:9 landscape
    (1080, 1920),  # 9:16 portrait
    (768, 1024),   # 3:4 portrait
    (1024, 768),   # 4:3 landscape
]

print("Resolution comparison for different input sizes:")
print("=" * 80)

for orig_w, orig_h in test_sizes:
    orig_pixels = orig_w * orig_h
    
    # Optimal Qwen resolution
    opt_w, opt_h = get_optimal_resolution(orig_w, orig_h)
    opt_pixels = opt_w * opt_h
    opt_ratio = opt_pixels / orig_pixels
    
    # Standard 1M pixel resolution
    std_w, std_h = get_standard_resolution(orig_w, orig_h)
    std_pixels = std_w * std_h
    std_ratio = std_pixels / orig_pixels
    
    print(f"\nOriginal: {orig_w}x{orig_h} ({orig_pixels:,} pixels)")
    print(f"  Optimal (Qwen): {opt_w}x{opt_h} ({opt_pixels:,} pixels) - {opt_ratio:.2f}x scale")
    print(f"  Standard (1M):  {std_w}x{std_h} ({std_pixels:,} pixels) - {std_ratio:.2f}x scale")
    
    # Calculate latent dimensions (assuming 8x downscale for VAE)
    opt_latent_h, opt_latent_w = opt_h // 8, opt_w // 8
    std_latent_h, std_latent_w = std_h // 8, std_w // 8
    
    print(f"  Optimal latent: {opt_latent_w}x{opt_latent_h}")
    print(f"  Standard latent: {std_latent_w}x{std_latent_h}")
    
    # Check if dimensions differ significantly
    if abs(opt_pixels - std_pixels) > 100000:
        print(f"  ⚠️  Large pixel difference: {abs(opt_pixels - std_pixels):,}")

print("\n" + "=" * 80)
print("\nKey observations:")
print("1. Optimal resolution snaps to predefined Qwen sizes")
print("2. Standard resolution maintains aspect ratio while targeting 1M pixels")
print("3. Different resolutions create different-sized reference latents")
print("4. Larger reference latents may have stronger influence on generation")