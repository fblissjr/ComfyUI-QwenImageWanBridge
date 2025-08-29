# Multi-Reference Image Ordering and Spatial Interpretation

## The Problem

When using `concat` or `grid` methods with multiple reference images, Qwen interprets the combined image spatially:
- **Left = First, Right = Second** (in concat mode)
- **Top-Left = First, Top-Right = Second** (in grid mode)

This can cause confusion when your prompt says "first image" or "second image".

## Understanding How Qwen Interprets Concatenated Images

**Setup:**
- image1: Coffee mug on table → Appears LEFT in concat
- image2: Plush doll on rainbow → Appears RIGHT in concat
- Method: concat creates: [Coffee Mug | Plush Doll]

**Important:** In concat mode:
- image1 = LEFT side = "first image" when referenced spatially
- image2 = RIGHT side = "second image" when referenced spatially

**Prompt Clarity Matters:**
- Ambiguous: "Place the object from the first image into the second image"
- Clear: "Take the object from the left and place it in the scene on the right"
- Clearest: "Put the coffee mug into the rainbow scene"

## Solutions

### Option 1: Use `index` Method (Recommended)
The `index` method keeps images truly separate and handles positional references better:
```
Method: index
Prompt: "Place the object from the first image into the scene of the second image"
```

### Option 2: Adjust Your Prompt for Spatial Layout
When using `concat`, refer to positions instead of order:
```
Method: concat  
Prompt: "Place the object from the LEFT image into the scene from the RIGHT image"
```

### Option 3: Swap Input Connections
Connect your images in reverse order:
- image1: Plush doll (will appear left)
- image2: Coffee mug (will appear right)
- Prompt: "Place the object from the right into the left scene"

## Visual Layout Reference

### Concat Method
```
[image1 | image2]
 LEFT     RIGHT
```

### Grid Method
```
[image1 | image2]
[image3 | image4]

TOP-LEFT  TOP-RIGHT
BOT-LEFT  BOT-RIGHT
```

## Best Practices

1. **For "first/second" references**: Use `index` method
2. **For style transfer**: Use `offset` method with weights
3. **For spatial comparison**: Use `concat` with left/right references
4. **For multiple variations**: Use `grid` with corner references

## Technical Details

The concatenation happens at the tensor level:
- `concat`: `torch.cat(images, dim=2)` - Horizontal concatenation
- `grid`: Creates 2x2 layout with images in reading order

Qwen's vision transformer processes images like text - left to right, top to bottom. This is why spatial position matters more than input order for concat/grid methods.