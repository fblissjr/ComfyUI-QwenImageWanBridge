# Z-Image Nodes

Text encoding nodes for Z-Image with experimental knobs for testing.

**Note:** After analysis, we found that ComfyUI and diffusers produce identical templates by default. Our nodes now match diffusers exactly, with optional experimental parameters for testing alternative approaches.

---

## ZImageTextEncoder

**Category:** `ZImage/Encoding`

Full-featured encoder with system prompts, templates, and debug mode. Default behavior matches diffusers exactly.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `clip` | CLIP | Yes | - | Z-Image CLIP model (lumina2 type) |
| `text` | STRING | Yes | "" | Your prompt - describe the image you want |
| `system_prompt_preset` | dropdown | No | "none" | Preset system prompts |
| `custom_system_prompt` | STRING | No | "" | Custom system prompt (overrides preset) |
| `template_preset` | dropdown | No | "none" | Template from `nodes/templates/z_image_*.md` |
| `add_think_block` | BOOLEAN | No | **False** | EXPERIMENTAL: Add `<think></think>` block |
| `thinking_content` | STRING | No | "" | EXPERIMENTAL: Custom reasoning inside `<think>` tags |
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

1. **Formats prompt** - Wraps text in Qwen3's chat template
2. **Applies system prompt** - Optionally prepends system instructions
3. **Optionally adds think block** - When `add_think_block=True` (experimental)
4. **Encodes via ComfyUI** - Uses ComfyUI's CLIP tokenizer and encoder
5. **Warns on long prompts** - Alerts if exceeding max_sequence_length

### Input Priority

System prompt is determined by (first match wins):

```
1. custom_system_prompt (if provided)  - Manual override
2. template_preset (if not "none")     - Template from file
3. system_prompt_preset (if not "none") - Built-in preset
4. (none)                              - No system prompt (diffusers default)
```

### Examples

**Default (matches diffusers):**
```
text: "A serene mountain lake at sunset with reflections"
add_think_block: False
system_prompt_preset: none
```

**With System Prompt:**
```
text: "Professional headshot of a woman in business attire"
system_prompt_preset: photorealistic
add_think_block: False
```

**Experimental: With Think Block:**
```
text: "A cyberpunk cityscape at night"
add_think_block: True
```

**Experimental: With Custom Thinking Content:**
```
text: "A golden retriever playing in autumn leaves"
add_think_block: True
thinking_content: "Key elements: happy dog, vibrant fall colors.
Composition: dog as focal point, leaves in mid-air.
Lighting: warm golden hour, backlit for rim lighting."
```

---

## ZImageTextEncoderSimple

**Category:** `ZImage/Encoding`

Drop-in replacement for CLIPTextEncode. Default behavior matches diffusers exactly.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `clip` | CLIP | Yes | - | Z-Image CLIP model (lumina2 type) |
| `text` | STRING | Yes | "" | Your prompt |
| `add_think_block` | BOOLEAN | No | **False** | EXPERIMENTAL: Add `<think></think>` block |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `conditioning` | CONDITIONING | Encoded text embeddings |

### How It Works

**With add_think_block=False (default, matches diffusers):**
```
<|im_start|>user
{your prompt}<|im_end|>
<|im_start|>assistant
```

**With add_think_block=True (experimental):**
```
<|im_start|>user
{your prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

### When to Use

- **Default behavior** matches both ComfyUI and diffusers exactly
- **Enable add_think_block** to experiment (may or may not improve results)
- **Use Full encoder** if you need system prompts or templates

---

## Understanding the Parameters

### add_think_block (EXPERIMENTAL)

This parameter is counterintuitively named in the original Qwen3 implementation:

| Qwen3's Name | Our Name | Effect |
|--------------|----------|--------|
| `enable_thinking=True` | `add_think_block=False` | NO think block (default) |
| `enable_thinking=False` | `add_think_block=True` | ADD think block |

We renamed to `add_think_block` for clarity - when True, it adds the block.

### Why Default is False

After testing, we found:
1. **Diffusers uses `enable_thinking=True`** which does NOT add think tokens
2. **ComfyUI's hardcoded template** also has no think tokens
3. **They produce identical output**

So the "fix" is to match diffusers default, which happens to be no think block.

### Tokenizer Caveat

Even with `add_think_block=True`, ComfyUI's bundled tokenizer doesn't have `<think>` as a special token:
- **Qwen3-4B tokenizer**: `<think>` = single token [151667]
- **ComfyUI tokenizer**: `<think>` = subwords `['<th', 'ink', '>']`

This means the think block is tokenized differently than intended.

---

## Available Templates (4 total)

Templates stored in `nodes/templates/z_image_*.md`:

| Template | Category | Description |
|----------|----------|-------------|
| `z_image_default` | general | No system prompt (matches diffusers default) |
| `z_image_photorealistic` | photography | Natural lighting, textures, realistic details |
| `z_image_bilingual_text` | text | English/Chinese text rendering |
| `z_image_artistic` | art | Creative compositions, visual balance |

---

## Debug Mode

Enable `debug_mode=True` on ZImageTextEncoder to see:

- System prompt source (preset, template, custom, or none)
- Whether think block is added
- Formatted prompt length in characters
- Estimated token count vs max_sequence_length
- Actual conditioning tensor shape
- Full formatted prompt text

### Example Debug Output

```
Using preset: photorealistic
add_think_block: False
max_sequence_length: 512
Formatted prompt length: 245 chars
Conditioning shape: torch.Size([1, 68, 2560])
Actual sequence length: 68

--- FORMATTED PROMPT ---
<|im_start|>system
Generate a photorealistic image...<|im_end|>
<|im_start|>user
A professional headshot of a woman<|im_end|>
<|im_start|>assistant
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
| Default matches diffusers | Yes | Yes |
| add_think_block option | Yes | Yes |
| System prompts | No | Yes (5 presets) |
| Custom system prompt | No | Yes |
| Template files | No | Yes (4 templates) |
| Custom thinking content | No | Yes |
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

**Default (add_think_block=False):**
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

**Experimental (add_think_block=True):**
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

---

## Tips

1. **Use defaults first** - They match diffusers exactly
2. **Experiment with think block** - May or may not help, test it
3. **Watch prompt length** - Debug mode shows if exceeding 512 tokens
4. **Compare outputs** - Same prompt, same seed, with/without think block
5. **Check debug output** - Verify your template/system prompt is being applied

---

## File Locations

- **Our encoder**: `nodes/z_image_encoder.py`
- **Templates**: `nodes/templates/z_image_*.md`
- **Documentation**: `nodes/docs/z_image_*.md`
- **ComfyUI's encoder**: `comfy/text_encoders/z_image.py`

---

**Last Updated:** 2025-11-27
**Version:** 2.0 (corrected parameter naming)
