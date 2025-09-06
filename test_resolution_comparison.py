#!/usr/bin/env python3
"""
Compare resolution calculations between native ComfyUI and our implementation
"""

import math

def comfyui_native_resolution(width, height):
    """ComfyUI's native TextEncodeQwenImageEdit resolution calculation"""
    total = int(1024 * 1024)
    scale_by = math.sqrt(total / (width * height))
    new_width = round(width * scale_by)
    new_height = round(height * scale_by)
    return new_width, new_height

def our_optimal_resolution(width, height):
    """Our optimal resolution finder"""
    QWEN_RESOLUTIONS = [
        (256, 256), (256, 512), (256, 768), (256, 1024), (256, 1280), (256, 1536), (256, 1792),
        (512, 256), (512, 512), (512, 768), (512, 1024), (512, 1280), (512, 1536), (512, 1792),
        (768, 256), (768, 512), (768, 768), (768, 1024), (768, 1280), (768, 1536),
        (1024, 256), (1024, 512), (1024, 768), (1024, 1024), (1024, 1280), (1024, 1536),
        (1280, 256), (1280, 512), (1280, 768), (1280, 1024), (1280, 1280),
        (1536, 256), (1536, 512), (1536, 768), (1536, 1024),
        (1792, 256), (1792, 512)
    ]
    
    target_pixels = width * height
    aspect_ratio = width / height
    
    # Find closest resolution by both pixel count AND aspect ratio
    best_res = min(
        QWEN_RESOLUTIONS,
        key=lambda r: abs(r[0] * r[1] - target_pixels) * 0.5 + 
                     abs((r[0] / r[1]) - aspect_ratio) * target_pixels * 0.5
    )
    
    return best_res

# Test cases
test_cases = [
    (512, 512),    # Square
    (1920, 1080),  # 16:9
    (1080, 1920),  # 9:16 portrait
    (800, 600),    # 4:3
    (1024, 768),   # 4:3
    (2048, 2048),  # Large square
    (640, 480),    # Small 4:3
    (1280, 720),   # 720p
]

print("Resolution Comparison")
print("=" * 80)
print(f"{'Input':<15} {'Native ComfyUI':<20} {'Our Optimal':<20} {'Difference':<20}")
print("-" * 80)

for width, height in test_cases:
    native_w, native_h = comfyui_native_resolution(width, height)
    optimal_w, optimal_h = our_optimal_resolution(width, height)
    
    native_pixels = native_w * native_h
    optimal_pixels = optimal_w * optimal_h
    pixel_diff = optimal_pixels - native_pixels
    
    print(f"{width}x{height:<8} {native_w}x{native_h:<13} {optimal_w}x{optimal_h:<13} {pixel_diff:+,} pixels")

print("\nKey Differences:")
print("- Native: Always scales to exactly 1,048,576 pixels (1024x1024)")
print("- Optimal: Snaps to nearest Qwen-supported resolution")
print("\nPotential Issues:")
print("1. Optimal may use MORE pixels than native (e.g., 1280x1024 = 1,310,720)")
print("2. Optimal may change aspect ratio slightly")
print("3. Different pixel counts can affect generation quality/behavior")