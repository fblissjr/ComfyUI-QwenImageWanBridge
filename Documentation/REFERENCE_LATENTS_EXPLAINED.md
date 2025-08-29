# Reference Latents in Qwen-Image-Edit: Complete Guide

## What Are Reference Latents?

Reference latents are encoded representations of your input image that guide the generation process. When you edit an image, Qwen needs to understand and preserve certain aspects of the original while making changes. Reference latents provide this "memory" of the original image.

## Reference Methods Explained

### 1. **"default" Method**
**How it works**: Standard injection into the model's conditioning
```python
# Injected once at the beginning
conditioning["reference_latents"] = [edit_latent]
```

**Behavior**:
- The model receives the reference as part of its context
- Applied uniformly throughout generation
- Model decides how much to use it

**When to use**:
- General editing tasks
- When you trust the model's judgment
- First option to try

**Example results**:
- Input: Photo of a red car
- Edit: "make it blue"
- Result: Blue car with same shape, position, background

---

### 2. **"inject" Method**
**How it works**: Forcefully injects reference at each denoising step
```python
# At each step:
noise = noise + (reference_latent * strength * (1 - sigma))
```

**Behavior**:
- Reference is actively mixed into the noise at each step
- Stronger preservation of original structure
- Can sometimes be too rigid

**When to use**:
- Need strong structural preservation
- Small, precise edits
- When default is changing too much

**Example results**:
- Input: Portrait photo
- Edit: "add glasses"
- Result: Same face, same expression, just adds glasses

---

### 3. **"concat" Method**
**How it works**: Concatenates reference with the noise latent
```python
# Before model processing:
latent = torch.cat([noise, reference_latent * strength], dim=1)
```

**Behavior**:
- Reference becomes part of the input channels
- Model learns to blend them
- More flexible than inject

**When to use**:
- Style transfer tasks
- Major changes while keeping composition
- When you want influence throughout

**Example results**:
- Input: Daytime city photo
- Edit: "make it night"
- Result: Same buildings, same layout, different lighting/mood

---

### 4. **"cross_attn" Method**
**How it works**: Uses cross-attention mechanism
```python
# In attention layers:
attention_output = CrossAttention(
    query=current_features,
    key=reference_features * strength,
    value=reference_features
)
```

**Behavior**:
- Most sophisticated method
- Reference guides through attention mechanism
- Selective influence based on relevance

**When to use**:
- Complex edits requiring intelligence
- When preserving specific details
- Professional/production work

**Example results**:
- Input: Person in park
- Edit: "change season to winter"
- Result: Same person, same pose, park becomes snowy

---

## Reference Strength Explained

Reference strength controls how much influence the original image has on the generation.

### Strength Values and Effects

#### **0.0 - No Reference**
- Completely ignores original image
- Pure text-to-image generation
- Loses all structure

**Example**:
- Input: Photo of a house
- Edit: "add a garden"
- Result: Completely new house with garden (not your house)

#### **0.1-0.3 - Minimal Reference**
- Very loose guidance
- Major changes possible
- Keeps only basic composition

**Example**:
- Input: Portrait photo
- Edit: "make it an oil painting"
- Result: Oil painting of a different person in similar pose

#### **0.4-0.6 - Moderate Reference**
- Balanced preservation and change
- Keeps main elements
- Allows style changes

**Example**:
- Input: Modern car
- Edit: "make it vintage"
- Result: Vintage car with similar shape/angle

#### **0.7-0.9 - Strong Reference**
- Preserves most details
- Only targeted changes
- High fidelity to original

**Example**:
- Input: Landscape photo
- Edit: "add sunset"
- Result: Exact same landscape with sunset lighting

#### **1.0 - Standard Reference**
- Default strength
- Model's intended balance
- **Recommended starting point**

**Example**:
- Input: Dog photo
- Edit: "make it a cat"
- Result: Cat in same pose/position as dog

#### **1.1-1.5 - Enhanced Reference**
- Stronger than intended
- Very conservative changes
- May resist edits

**Example**:
- Input: Red dress
- Edit: "make it blue"
- Result: Slightly blue-tinted red dress

#### **1.6-2.0 - Overpowering Reference**
- Can cause artifacts
- Fights against changes
- Usually too strong

**Example**:
- Input: Sunny day
- Edit: "make it rainy"
- Result: Sunny day with weird rain artifacts

---

## How Reference Latents Work Internally

### The Technical Process

1. **Image Encoding**
```python
edit_latent = vae.encode(input_image)  # 16-channel latent
```

2. **Reference Injection**
```python
# Added to conditioning
conditioning["reference_latents"] = [edit_latent * reference_strength]
conditioning["reference_method"] = reference_method
```

3. **Model Processing**
```python
# In the model's forward pass
if "reference_latents" in conditioning:
    method = conditioning["reference_method"]
    if method == "inject":
        x = x + reference_latents[0] * (1 - timestep/1000)
    elif method == "concat":
        x = torch.cat([x, reference_latents[0]], dim=1)
    elif method == "cross_attn":
        x = cross_attention(x, reference_latents[0])
```

4. **Progressive Influence**
- Early steps: Structure establishment
- Middle steps: Detail refinement  
- Late steps: Fine details

---

## Practical Combinations

### Best Method + Strength Combinations

#### **Subtle Edits**
- Method: `inject`
- Strength: `0.8-1.2`
- Example: "fix lighting", "remove blemishes"

#### **Style Transfer**
- Method: `concat`
- Strength: `0.5-0.7`
- Example: "make it anime style", "convert to watercolor"

#### **Object Replacement**
- Method: `cross_attn`
- Strength: `0.6-0.8`
- Example: "replace car with truck", "change dog breed"

#### **Major Scene Changes**
- Method: `default`
- Strength: `0.3-0.5`
- Example: "change season", "day to night"

#### **Preserving Identity**
- Method: `inject` or `cross_attn`
- Strength: `1.0-1.3`
- Example: "change clothing", "add accessories"

---

## Common Issues and Solutions

### Problem: "Changes nothing despite edit prompt"
**Cause**: Reference strength too high
**Solution**: Lower to 0.5-0.7

### Problem: "Loses all structure"
**Cause**: Reference strength too low or wrong method
**Solution**: Increase to 0.8-1.0, try `inject` method

### Problem: "Artifacts and distortions"
**Cause**: Reference strength >1.5
**Solution**: Keep below 1.3

### Problem: "Partial changes only"
**Cause**: Method fighting the edit
**Solution**: Try different method, adjust strength

---

## Advanced Tips

### 1. **Multi-Stage Editing**
```
Stage 1: Major changes (strength: 0.4, method: default)
Stage 2: Refinement (strength: 0.8, method: inject)
Stage 3: Details (strength: 1.2, method: cross_attn)
```

### 2. **Dynamic Strength**
- Start with low strength for initial structure
- Increase gradually for detail preservation
- Useful in autoregressive chains

### 3. **Method Switching**
- Use `concat` for first half of steps
- Switch to `inject` for second half
- Combines flexibility with precision

### 4. **Reference Masking** (Advanced)
```python
# Only use reference for certain regions
masked_reference = reference_latent * spatial_mask
```

---

## Quick Reference Guide

| Edit Type | Best Method | Typical Strength | Denoise |
|-----------|------------|------------------|---------|
| Color change | inject | 0.8-1.0 | 0.3-0.5 |
| Style transfer | concat | 0.5-0.7 | 0.6-0.8 |
| Object swap | cross_attn | 0.6-0.8 | 0.5-0.7 |
| Add elements | default | 0.7-0.9 | 0.4-0.6 |
| Remove elements | inject | 1.0-1.2 | 0.3-0.5 |
| Lighting change | concat | 0.6-0.8 | 0.4-0.6 |
| Expression change | cross_attn | 0.9-1.1 | 0.3-0.4 |
| Background swap | default | 0.4-0.6 | 0.6-0.8 |

---

## Summary

- **Reference Method**: HOW the original image influences generation
- **Reference Strength**: HOW MUCH influence it has
- **Start with**: method=`default`, strength=`1.0`
- **Adjust based on**: How much you want to preserve vs change
- **Remember**: Lower strength = more change, Higher = more preservation