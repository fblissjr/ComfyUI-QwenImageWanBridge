# Z-Image Text Encoder

**Category:** ZImage/Encoding
**Display Names:**
- Z-Image Text Encoder (full-featured)
- Z-Image Text Encode (Simple) (drop-in replacement)

## Overview

Custom encoder nodes for Z-Image that expose experimental parameters for testing. Z-Image uses Qwen3-4B as its text encoder.

**Key Finding:** After analysis, we found ComfyUI and diffusers produce **identical templates** by default. Our nodes match diffusers exactly, with optional experimental parameters.

## ComfyUI vs Diffusers: What We Found

### Template Format: IDENTICAL

**ComfyUI** (z_image.py):
```python
self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
```

**Diffusers** with `enable_thinking=True` (the default):
```python
tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=True  # Does NOT add think block!
)
# Result: "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
```

**Both produce the same template** - no think block.

### The `enable_thinking` Confusion

The Qwen3 parameter name is counterintuitive:

| Qwen3 Parameter | Template Result | Explanation |
|-----------------|-----------------|-------------|
| `enable_thinking=True` | NO `<think>` block | Allows model to generate thinking (for LLM generation) |
| `enable_thinking=False` | ADD `<think></think>` | Pre-fills empty block to skip thinking |

For text **encoding** (not generation), diffusers uses `enable_thinking=True`, which produces NO think block. ComfyUI's hardcoded template matches this exactly.

### The Real Difference: Embedding Extraction

| Aspect | Diffusers | ComfyUI |
|--------|-----------|---------|
| Template | No think block | No think block |
| Token IDs | Identical | Identical |
| Embedding extraction | Filters by attention mask (variable length) | Returns full padded sequence + mask |

**Diffusers** filters embeddings to valid tokens only:
```python
for i in range(len(prompt_embeds)):
    embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
```

**ComfyUI** returns full padded sequence (512 tokens) with attention mask in `extra` dict. The Z-Image model uses this mask during context refinement, but main transformer layers receive `mask=None`.

This difference **cannot be fixed** without modifying ComfyUI core.

## Nodes

### ZImageTextEncoder (Full-Featured)

Full encoder with system prompts, templates, and debug mode.

#### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| clip | CLIP | Yes | - | Z-Image CLIP model (lumina2 type) |
| text | STRING | Yes | "" | Your prompt |
| template_preset | ENUM | No | "none" | Template from `nodes/templates/z_image/` (auto-fills system_prompt) |
| system_prompt | STRING | No | "" | Editable system prompt (auto-filled by template via JS) |
| raw_prompt | STRING | No | "" | RAW MODE: Bypass all formatting, use your own tokens |
| add_think_block | BOOLEAN | No | **False** | Add `<think></think>` block (auto-enabled if thinking_content provided) |
| thinking_content | STRING | No | "" | Content INSIDE `<think>...</think>` tags |
| assistant_content | STRING | No | "" | Content AFTER `</think>` tags |
| max_sequence_length | INT | No | 512 | Maximum tokens (matches diffusers) |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| conditioning | CONDITIONING | Encoded text embeddings (2560 dimensions) |
| formatted_prompt | STRING | Exact prompt that was encoded (for debugging) |

### ZImageTextEncoderSimple (Drop-in Replacement)

Minimal encoder - drop-in replacement for CLIPTextEncode. Default behavior matches diffusers exactly.

#### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| clip | CLIP | Yes | - | Z-Image CLIP model |
| text | STRING | Yes | "" | Your prompt |
| raw_prompt | STRING | No | "" | RAW: Bypass formatting, use your own tokens |
| add_think_block | BOOLEAN | No | **False** | Add `<think></think>` block (auto-enabled if thinking_content provided) |
| thinking_content | STRING | No | "" | Content INSIDE `<think>...</think>` tags |
| assistant_content | STRING | No | "" | Content AFTER `</think>` tags |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| conditioning | CONDITIONING | Encoded text embeddings |
| formatted_prompt | STRING | Exact prompt that was encoded |

## Qwen3 Model Variants

As of the 2507 release, there are three Qwen3 model types:

| Variant | Thinking Mode | Notes |
|---------|---------------|-------|
| **Qwen3-4B** (2504 hybrid) | Switchable | Z-Image uses this |
| Qwen3-Instruct-2507 | Never | Parameter not supported |
| Qwen3-Thinking-2507 | Always | Always thinks |

**Z-Image uses the 2504 hybrid model** (Qwen3-4B without suffix).

## Known Gaps vs Diffusers

### Gap 1: Embedding Extraction (Cannot Fix)

**Diffusers** extracts only valid (non-padded) tokens:
```python
for i in range(len(prompt_embeds)):
    embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
```

**ComfyUI** returns the full padded sequence. The attention mask is passed to the model and used during context refinement, but the sequence length differs.

**Why we can't fix this**: ComfyUI's CLIP architecture expects fixed-shape tensors. Changing this would require modifying ComfyUI core.

**Impact**: RoPE position IDs for image patches differ (they start after caption length). May affect quality subtly.

### Gap 2: Sequence Length (Fixed)

**Diffusers** uses `max_sequence_length=512` with truncation.

**ComfyUI** uses effectively unlimited length.

**Our fix**: Added `max_sequence_length` parameter (default: 512) with warning when exceeded.

### Gap 3: Tokenizer Special Tokens (Minor)

ComfyUI bundles a Qwen2.5-VL tokenizer where token 151667 = `<|meta|>`.
Qwen3-4B tokenizer has token 151667 = `<think>`.

When `add_think_block=True`, ComfyUI tokenizes `<think>` as subwords `['<th', 'ink', '>']` instead of a single special token.

## Template Files

Templates stored in `nodes/templates/z_image/` subfolder:

| Template | Description |
|----------|-------------|
| `default` | No system prompt (matches diffusers default) |
| `photorealistic` | Photography with natural lighting |
| `bilingual_text` | English/Chinese text rendering |
| `artistic` | Creative compositions |
| ... | Many more templates available (see folder) |

## Recommended Workflow

### Basic (matches diffusers)

```
CLIPLoader (lumina2) --> ZImageTextEncoderSimple --> KSampler
                              |
                         add_think_block=False (default)
```

### With System Prompts

```
CLIPLoader (lumina2) --> ZImageTextEncoder --> KSampler
                              |
                         system_prompt_preset="photorealistic"
                         add_think_block=False
```

### Experimental: With Think Block

```
CLIPLoader (lumina2) --> ZImageTextEncoderSimple --> KSampler
                              |
                         add_think_block=True
```

Test if adding think block helps or hurts quality.

## Technical Details

### Qwen3-4B Specifications

| Parameter | Value |
|-----------|-------|
| Hidden Size | 2560 |
| Layers | 36 |
| Vocabulary | 151,936 |
| Max Position Embeddings | 40,960 |
| Embedding Layer Used | `hidden_states[-2]` (second-to-last) |
| dtype | bfloat16 |

### Template Format

**Default (add_think_block=False, matches diffusers):**
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

## File Locations

- **Our encoder**: `nodes/z_image_encoder.py`
- **ComfyUI's encoder**: `comfy/text_encoders/z_image.py`
- **Diffusers pipeline**: `diffusers/pipelines/z_image/pipeline_z_image.py`
- **Templates**: `nodes/templates/z_image/` (subfolder with `.md` files)
- **JS auto-fill**: `web/js/z_image_encoder.js`
- **API endpoint**: `/api/z_image_templates` (registered in `__init__.py`)

## Related Documentation

- [Z-Image Nodes Reference](z_image_nodes.md) - Detailed node documentation
- [Z-Image Workflow Guide](z_image_workflow_guide.md) - Step-by-step setup guide
- [Z-Image Analysis](z_image_analysis.md) - Deep dive: ComfyUI vs Diffusers comparison
- [Z-Image Turbo Workflow Analysis](z_image_turbo_workflow_analysis.md) - Official workflow breakdown

---

**Last Updated:** 2025-11-28
**Version:** 2.1 (template subfolder, API endpoint, updated params)
