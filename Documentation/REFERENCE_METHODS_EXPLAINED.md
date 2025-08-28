# Reference Methods in QwenVLTextEncoder

## Current Status
**Single reference only** - The `reference_method` parameter exists but doesn't do anything useful yet since our node only accepts one image.

## What Reference Methods Are For

Reference methods control how multiple reference images are combined when passed to the model. This is a Flux-inspired feature for advanced workflows.

### The Three Methods

1. **standard** (default, current)
   - Single reference image
   - What we use now
   - Image gets encoded to reference latents

2. **index** (future feature)
   - Multiple images with positional encoding
   - Each image gets unique position in sequence
   - Example: [style_img][content_img][text_prompt]
   - Use case: "Apply style from image 1 to content of image 2"

3. **offset** (future feature)  
   - Multiple images stacked/combined in latent space
   - Creates blended reference
   - Use case: Mixing multiple style references

## How to Reference an Image's Style (Current)

With our current single-image limitation:

```
LoadImage (style reference) → QwenVLTextEncoder.edit_image
Prompt: "Transform this into [target] with the same artistic style"
```

The model understands style from vision tokens, so prompts like these work:
- "Keep the artistic style but change to a sunset"
- "Apply this painting style to a photograph of a cat"
- "Maintain the color palette and brushwork"

## What Multi-Reference Would Enable (Future)

If we supported multiple images:

### Style Transfer
```
Image 1: Artistic style reference
Image 2: Content/structure reference  
Prompt: "Combine the style of the first image with content of the second"
Reference Method: index
```

### Multiple Style Blending
```
Image 1: Watercolor style
Image 2: Impressionist style
Prompt: "Blend these artistic styles"
Reference Method: offset
```

### Context + Edit
```
Image 1: Original scene
Image 2: Object to add
Prompt: "Add the second image's object into the first scene"
Reference Method: index
```

## Why We Don't Support This Yet

1. **ComfyUI Architecture**: Expects single image inputs
2. **Node Complexity**: Would need list/batch image handling
3. **Limited Use Cases**: Most edits only need one reference
4. **Native Support**: ComfyUI doesn't have multi-reference infrastructure

## Workaround for Multiple References

You can describe multiple images in your prompt:
```
"Create an image combining Van Gogh's Starry Night swirls with Monet's water lily colors"
```

The model has learned these styles during training, so verbal descriptions often work well.

## Bottom Line

- Keep `reference_method` as "standard" 
- It's a placeholder for future multi-image support
- For style reference, use descriptive prompts with your single reference image
- Most workflows don't need multiple references anyway