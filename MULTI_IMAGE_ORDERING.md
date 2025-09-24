# Multi-Image Ordering Guide

## How Images Are Numbered

When using multi-image editing with Qwen-Image-Edit-2509, images are numbered based on the order they're connected to the Image Batch node.

## Visual Example

```
LoadImage (person.jpg)     → image_1 input → Image Batch → QwenVLTextEncoder
LoadImage (background.jpg) → image_2 input → Image Batch →
LoadImage (style_ref.jpg)  → image_3 input → Image Batch →
```

In your prompts, these become:
- **Picture 1**: person.jpg
- **Picture 2**: background.jpg
- **Picture 3**: style_ref.jpg

## Using Image Batch (KJNodes)

The Image Batch node from KJNodes has numbered inputs:
- `image_1` → Picture 1 in your prompt
- `image_2` → Picture 2 in your prompt
- `image_3` → Picture 3 in your prompt
- etc.

Set the `inputcount` widget to match the number of images you're using.

## Example Prompts

Once you know the order:
- "Take the old man from Picture 1 and place him in the scene from Picture 2"
- "Combine the subject from Picture 1 with the background from Picture 2 using the style of Picture 3"
- "Replace the person in Picture 2 with the person from Picture 1"

## Debugging Image Order

Enable `debug_mode` in QwenVLTextEncoder to see:
```
[Encoder] Applied image_edit template for 3 images
[Encoder] Full formatted text being tokenized: Picture 1: <|vision_start|><|image_pad|><|vision_end|>Picture 2: ...
```

This shows you exactly how the encoder is formatting your images.

## Tips

1. **Name your LoadImage nodes** - Right-click → Properties → Title
   - Name them "Image 1: Person", "Image 2: Background", etc.

2. **Preview each image** - Right-click LoadImage → Preview to verify

3. **Use debug mode** - Shows the Picture X formatting being applied

4. **Test with simple prompts first**:
   - "Show Picture 1" - Should emphasize first image
   - "Show Picture 2" - Should emphasize second image

## Common Workflow

1. Load your images with LoadImage nodes
2. Connect them IN ORDER to Image Batch inputs
3. Note which image connects to which input
4. Write your prompt using "Picture 1", "Picture 2", etc.
5. Enable debug_mode to verify formatting

## Visual Workspace Organization

Arrange your nodes vertically to match the order:
```
┌─────────────────┐
│ LoadImage       │ ← Picture 1
│ (person.jpg)    │
└────────┬────────┘
         │
┌────────▼────────┐
│ LoadImage       │ ← Picture 2
│ (background.jpg)│
└────────┬────────┘
         │
┌────────▼────────┐
│ LoadImage       │ ← Picture 3
│ (style.jpg)     │
└────────┬────────┘
         │
┌────────▼────────┐
│ Image Batch     │
└─────────────────┘
```

This makes it visually clear which image is which Picture number.