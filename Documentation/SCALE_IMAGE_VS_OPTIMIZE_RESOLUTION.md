# Scale Image Node vs optimize_resolution: Which to Use?

## TL;DR
**Remove the Scale Image to Total Pixels node from your workflows.** The QwenVLTextEncoder's built-in resolution handling is better and avoids quality-degrading double scaling.

## The Problem: Redundant Double Scaling

In our original workflows, we had:
```
LoadImage → Scale Image (1M pixels) → QwenVLTextEncoder (scales again!) → ...
```

This causes the image to be scaled TWICE:
1. First by Scale Image node to 1 megapixel
2. Again inside the encoder (when optimize_resolution=False) to 1 megapixel

**Result**: Unnecessary quality loss from resampling the image twice.

## The Solution: Use Built-in Resolution Handling

Connect directly:
```
LoadImage → QwenVLTextEncoder → ...
```

Then choose your resolution strategy:

### Option 1: Standard Scaling (optimize_resolution=False)
- Scales to 1024×1024 total pixels (1 megapixel)
- Simple, predictable behavior
- Same result as the old Scale Image node, but only scales once

### Option 2: Qwen-Optimized Resolutions (optimize_resolution=True) 
- Snaps to nearest Qwen-supported resolution
- 36 official resolutions the model was trained on
- Potentially better quality since model knows these exact dimensions
- Examples: 1024×1024, 1280×768, 768×1280, etc.

## Migration Guide

### Old Workflow (Remove This):
1. LoadImage
2. Scale Image to Total Pixels (1 megapixel)
3. QwenVLTextEncoder (optimize_resolution=False)

### New Workflow (Use This):
1. LoadImage
2. QwenVLTextEncoder (optimize_resolution=True recommended)

### Why optimize_resolution=True is Recommended:
- Uses exact resolutions Qwen was trained on
- Better spatial understanding 
- No quality loss from arbitrary scaling
- Automatic aspect ratio preservation

## FAQ

**Q: What if I want a different target size?**
A: The encoder always targets ~1M pixels for optimal model performance. This is what Qwen was trained on.

**Q: Can I still use Scale Image for other purposes?**
A: Yes, but not for Qwen preprocessing. Use it for other workflows that need specific dimensions.

**Q: Which interpolation method is used?**
A: The encoder uses lanczos interpolation, same as the Scale Image node.

**Q: Will removing Scale Image break my existing workflows?**
A: No, just reconnect LoadImage directly to QwenVLTextEncoder and set optimize_resolution=True.

## Performance Comparison

| Method | Scales | Quality | Speed |
|--------|--------|---------|-------|
| Scale Image + Encoder (optimize=False) | 2× | Degraded (double resampling) | Slower |
| Encoder only (optimize=False) | 1× | Good | Fast |
| Encoder only (optimize=True) | 1× | Best (trained resolutions) | Fast |

## Bottom Line

The Scale Image to Total Pixels node is now obsolete for Qwen workflows. The encoder's built-in resolution handling is:
- More efficient (single scaling operation)
- Higher quality (no double resampling)
- More intelligent (can use Qwen's trained resolutions)
- Simpler (one less node to configure)

Always connect LoadImage directly to QwenVLTextEncoder and use optimize_resolution=True for best results.