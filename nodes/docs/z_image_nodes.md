# Z-Image Nodes

Custom text encoding nodes for Z-Image that fix ComfyUI's missing thinking tokens.

---

## ZImageTextEncoder

**Category:** `ZImage/Encoding`

Full-featured encoder with system prompts, templates, and debug mode. Fixes ComfyUI's missing thinking tokens to match diffusers implementation.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `clip` | CLIP | Yes | - | Z-Image CLIP model (lumina2 type) |
| `text` | STRING | Yes | "" | Your prompt - describe the image you want |
| `system_prompt_preset` | dropdown | No | "none" | Preset system prompts |
| `custom_system_prompt` | STRING | No | "" | Custom system prompt (overrides preset) |
| `template_preset` | dropdown | No | "none" | Template from `nodes/templates/z_image_*.md` |
| `enable_thinking` | BOOLEAN | No | True | Add thinking tokens (recommended) |
| `max_sequence_length` | INT | No | 512 | Max tokens (matches diffusers default) |
| `debug_mode` | BOOLEAN | No | False | Show encoding details |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `conditioning` | CONDITIONING | Encoded text embeddings (2560 dimensions) |
| `debug_output` | STRING | Debug info (when debug_mode=True) |

### System Prompt Presets

| Preset | Description |
|--------|-------------|
| `none` | No system prompt (matches diffusers default) |
| `quality` | High-quality, detailed image with excellent composition |
| `photorealistic` | Photorealistic with accurate lighting, textures, natural details |
| `artistic` | Artistic, aesthetically pleasing with creative composition |
| `bilingual` | Accurate text rendering for English and Chinese |

### How It Works

1. **Formats prompt with thinking tokens** - Wraps your text in Qwen3's chat template with `<think>` tokens
2. **Applies system prompt** - Optionally prepends system instructions for style guidance
3. **Encodes via ComfyUI** - Uses ComfyUI's CLIP tokenizer and encoder
4. **Warns on long prompts** - Alerts if prompt likely exceeds max_sequence_length

### Input Priority

System prompt is determined by (first match wins):

```
1. custom_system_prompt (if provided)  - Manual override
2. template_preset (if not "none")     - Template from file
3. system_prompt_preset (if not "none") - Built-in preset
4. (none)                              - No system prompt (diffusers default)
```

### Examples

**Basic (with thinking fix):**
```
text: "A serene mountain lake at sunset with reflections"
enable_thinking: True
system_prompt_preset: none
```

**With System Prompt:**
```
text: "Professional headshot of a woman in business attire"
system_prompt_preset: photorealistic
enable_thinking: True
```

**With Custom System Prompt:**
```
text: "A cyberpunk cityscape at night"
custom_system_prompt: "Generate a highly detailed sci-fi image with neon lighting, rain-slicked streets, and holographic advertisements."
enable_thinking: True
```

**With Template File:**
```
text: "A vintage photograph of Paris in the 1920s"
template_preset: z_image_artistic
enable_thinking: True
```

---

## ZImageTextEncoderSimple

**Category:** `ZImage/Encoding`

Minimal encoder - just adds the missing thinking tokens. Drop-in replacement for CLIPTextEncode when using Z-Image.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `clip` | CLIP | Yes | - | Z-Image CLIP model (lumina2 type) |
| `text` | STRING | Yes | "" | Your prompt |
| `enable_thinking` | BOOLEAN | No | True | Add thinking tokens (recommended) |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `conditioning` | CONDITIONING | Encoded text embeddings |

### How It Works

Simply wraps your prompt in the correct template:

**With enable_thinking=True:**
```
<|im_start|>user
{your prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

**With enable_thinking=False (ComfyUI default):**
```
<|im_start|>user
{your prompt}<|im_end|>
<|im_start|>assistant
```

### When to Use

- **Use Simple** when you just want the thinking token fix without any system prompts
- **Use Full** when you want to experiment with system prompts, templates, or debugging

---

## Available Templates (4 total)

Templates stored in `nodes/templates/z_image_*.md`:

| Template | Category | Description |
|----------|----------|-------------|
| `z_image_default` | general | No system prompt (matches diffusers default) |
| `z_image_photorealistic` | photography | Natural lighting, textures, realistic details |
| `z_image_bilingual_text` | text | English/Chinese text rendering |
| `z_image_artistic` | art | Creative compositions, visual balance |

### Template File Format

Templates use YAML frontmatter:

```markdown
---
name: z_image_photorealistic
description: Photorealistic image generation
model: z-image
category: photography
---
Generate a photorealistic image with accurate lighting, natural textures, and realistic details.
```

### Adding Custom Templates

1. Create `nodes/templates/z_image_yourtemplate.md`
2. Add YAML frontmatter with required fields
3. Add your system prompt as the body
4. Restart ComfyUI to load

---

## Debug Mode

Enable `debug_mode=True` on ZImageTextEncoder to see:

- System prompt source (preset, template, custom, or none)
- Whether thinking tokens are enabled
- Formatted prompt length in characters
- Estimated token count vs max_sequence_length
- Actual conditioning tensor shape
- Full formatted prompt text

### Example Debug Output

```
Using preset: photorealistic
enable_thinking: True
max_sequence_length: 512
Formatted prompt length: 287 chars
Conditioning shape: torch.Size([1, 72, 2560])
Actual sequence length: 72

--- FORMATTED PROMPT ---
<|im_start|>system
Generate a photorealistic image with accurate lighting, natural textures, and realistic details.<|im_end|>
<|im_start|>user
A professional headshot of a woman<|im_end|>
<|im_start|>assistant
<think>

</think>

--- END ---
```

---

## Basic Workflow

```
CLIPLoader (lumina2)
        |
        | CLIP
        v
ZImageTextEncoderSimple
        |
        | CONDITIONING
        v
KSampler <-- EmptySD3LatentImage
        |
        | LATENT
        v
VAEDecode <-- VAELoader
        |
        | IMAGE
        v
SaveImage
```

---

## Comparison: Simple vs Full

| Feature | ZImageTextEncoderSimple | ZImageTextEncoder |
|---------|------------------------|-------------------|
| Thinking tokens | Yes | Yes |
| System prompts | No | Yes (5 presets) |
| Custom system prompt | No | Yes |
| Template files | No | Yes (4 templates) |
| Max sequence length | No | Yes (configurable) |
| Debug mode | No | Yes |
| Drop-in replacement | Yes | No |

---

## Model Requirements

- **CLIP Model**: `qwen_3_4b.safetensors` loaded with `lumina2` type
- **DiT Model**: `z_image_turbo_bf16.safetensors` or similar
- **VAE**: `ae.safetensors` (Z-Image VAE)

### Model Paths

```
ComfyUI/models/
  text_encoders/
    qwen_3_4b.safetensors
  diffusion_models/
    z_image_turbo_bf16.safetensors
  vae/
    ae.safetensors
```

---

## Technical Details

### Qwen3-4B Specifications

| Parameter | Value |
|-----------|-------|
| Hidden Size | 2560 |
| Layers | 36 |
| Vocabulary | 151,936 |
| Max Position Embeddings | 40,960 |
| Embedding Layer | `hidden_states[-2]` |
| dtype | bfloat16 |

### Token Format

With `enable_thinking=True`, your prompt becomes:

```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The `<think>` block is empty but present - this matches how the model was trained.

### Why Thinking Tokens Matter

Qwen3-4B (no suffix) is the instruct model with hybrid thinking mode:
- Model was trained with `<think>` tokens in the template
- Omitting them puts embeddings out-of-distribution
- Adding them aligns with how diffusers encodes prompts

---

## Tips

1. **Start with Simple** - Use ZImageTextEncoderSimple first, only switch to Full if you need system prompts
2. **Keep thinking enabled** - `enable_thinking=True` is recommended for best quality
3. **Watch prompt length** - Debug mode shows if you're exceeding 512 tokens
4. **Compare outputs** - Test same prompt with/without thinking to verify improvement
5. **Check debug output** - Verify your template/system prompt is being applied

---

## Troubleshooting

### "Low quality output"

**Solution:** Ensure `enable_thinking=True` (the whole point of these nodes)

### "Different results than diffusers"

**Possible causes:**
- Embedding extraction differs (we return padded, diffusers filters)
- System prompt differences
- Sampling parameters (Z-Image Turbo uses steps=9, cfg=1)

### "Prompt seems truncated"

**Solution:**
- Enable debug mode to check actual sequence length
- Shorten prompt or increase max_sequence_length
- ComfyUI may handle long prompts differently than diffusers

### "System prompt not applied"

**Solution:**
- Check input priority (custom > template > preset)
- Verify template file exists and is valid YAML
- Enable debug mode to see which source is being used

---

## File Locations

- **Our encoder**: `nodes/z_image_encoder.py`
- **Templates**: `nodes/templates/z_image_*.md`
- **Documentation**: `nodes/docs/z_image_*.md`
- **ComfyUI's encoder**: `comfy/text_encoders/z_image.py` (has the bug)

---

**Last Updated:** 2025-11-27
