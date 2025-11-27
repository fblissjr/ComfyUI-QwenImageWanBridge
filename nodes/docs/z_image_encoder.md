# Z-Image Text Encoder

**Category:** ZImage/Encoding
**Display Names:**
- Z-Image Text Encoder (full-featured)
- Z-Image Text Encode (Simple) (drop-in replacement)

## Overview

Custom encoder nodes for Z-Image that fix ComfyUI's incorrect template handling. Z-Image uses Qwen3-4B as its text encoder, and ComfyUI's built-in implementation omits critical tokens that the model was trained with.

## The Problem: ComfyUI's Missing Thinking Tokens

### What ComfyUI Does (Incorrect)

ComfyUI hardcodes this template in `comfy/text_encoders/z_image.py`:

```python
self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
```

### What Diffusers Does (Correct)

Diffusers uses `apply_chat_template` with thinking mode enabled:

```python
prompt_item = self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # This is the key difference
)
```

This produces:

```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

### Why This Matters

- Qwen3-4B (no suffix) is the **instruct model**, not the base model
- The instruct model was trained with thinking mode support
- Missing `<think>\n\n</think>\n\n` tokens produce **out-of-distribution embeddings**
- This can cause subtle quality degradation in generated images

## Nodes

### ZImageTextEncoder (Full-Featured)

Full encoder with system prompts, templates, and debug mode.

#### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| clip | CLIP | Yes | Z-Image CLIP model (lumina2 type) |
| text | STRING | Yes | Your prompt |
| system_prompt_preset | ENUM | No | Preset system prompts: none, quality, photorealistic, artistic, bilingual |
| custom_system_prompt | STRING | No | Custom system prompt (overrides preset) |
| template_preset | ENUM | No | Template from `nodes/templates/z_image_*.md` |
| enable_thinking | BOOLEAN | No | Add thinking tokens (default: True, recommended) |
| max_sequence_length | INT | No | Maximum tokens (default: 512, matches diffusers) |
| debug_mode | BOOLEAN | No | Show encoding details |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| conditioning | CONDITIONING | Encoded text embeddings (2560 dimensions) |
| debug_output | STRING | Processing details when debug_mode=True |

### ZImageTextEncoderSimple (Drop-in Replacement)

Minimal encoder - just adds the missing thinking tokens. Use as a drop-in replacement for CLIPTextEncode when using Z-Image.

#### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| clip | CLIP | Yes | Z-Image CLIP model |
| text | STRING | Yes | Your prompt |
| enable_thinking | BOOLEAN | No | Add thinking tokens (default: True) |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| conditioning | CONDITIONING | Encoded text embeddings |

## Qwen3 Model Variants

As of the 2507 release, there are three Qwen3 model types:

| Variant | Thinking Mode | `enable_thinking` Support |
|---------|---------------|---------------------------|
| **Qwen3-4B** (2504 hybrid) | Switchable | Yes - use `True` for Z-Image |
| Qwen3-Instruct-2507 | Never | No - parameter not supported |
| Qwen3-Thinking-2507 | Always | N/A - always thinks |

**Z-Image uses the 2504 hybrid model** (Qwen3-4B without suffix), which supports thinking mode switching. Our nodes default to `enable_thinking=True` to match diffusers.

## Known Gaps vs Diffusers

### Gap 1: Embedding Extraction (Cannot Fix)

**Diffusers** extracts only valid (non-padded) tokens:
```python
for i in range(len(prompt_embeds)):
    embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
```

**ComfyUI** returns the full padded sequence including padding embeddings.

**Why we can't fix this**: ComfyUI's CLIP architecture expects fixed-shape tensors through `clip.encode_from_tokens_scheduled()`. Changing this would require modifying ComfyUI core, not just our nodes.

**Impact**: Semantic difference in what the DiT receives. May affect quality, but the thinking token fix is likely more important.

### Gap 2: Sequence Length (Fixed)

**Diffusers** uses `max_sequence_length=512` with truncation.

**ComfyUI** uses `max_length=99999999` (effectively unlimited).

**Our fix**: Added `max_sequence_length` parameter (default: 512) with truncation warning.

### Gap 3: Bundled Tokenizer Template (Cannot Fix)

ComfyUI bundles a Qwen2.5-style tokenizer config without Qwen3 thinking template support. This is why we manually construct the template with thinking tokens rather than using `apply_chat_template`.

## Template Files

Templates stored in `nodes/templates/z_image_*.md`:

| Template | Description |
|----------|-------------|
| `z_image_default` | No system prompt (matches diffusers default) |
| `z_image_photorealistic` | Photography with natural lighting |
| `z_image_bilingual_text` | English/Chinese text rendering |
| `z_image_artistic` | Creative compositions |

## Recommended Workflow

### Basic (with thinking fix)

```
CLIPLoader (lumina2) --> ZImageTextEncoderSimple --> KSampler
                              |
                         enable_thinking=True
```

### With System Prompts

```
CLIPLoader (lumina2) --> ZImageTextEncoder --> KSampler
                              |
                         system_prompt_preset="photorealistic"
                         enable_thinking=True
```

### Comparison Testing

To verify the thinking token fix helps:

1. Run with `enable_thinking=True` (our fix)
2. Run with `enable_thinking=False` (ComfyUI default behavior)
3. Same prompt, same seed, compare outputs

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

### Special Token IDs

| Token | ID | Purpose |
|-------|-----|---------|
| `<\|endoftext\|>` | 151643 | End of document / padding |
| `<\|im_end\|>` | 151645 | End of turn (eos) |
| `</think>` | 151668 | Thinking content delimiter |

### Template Format (What We Generate)

With `enable_thinking=True`:
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

With `enable_thinking=False` (matches ComfyUI default):
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

## Sampling Recommendations (from Qwen3 Official Docs)

While these are for generation (not embedding), they may inform future experiments:

| Setting | Thinking Mode | Non-Thinking Mode |
|---------|---------------|-------------------|
| Temperature | 0.6 | 0.7 |
| top_p | 0.95 | 0.8 |
| top_k | 20 | 20 |

**Important**: Qwen3 docs warn "DO NOT use greedy decoding" - though this applies to generation, not embedding extraction.

## File Locations

- **Our encoder**: `nodes/z_image_encoder.py`
- **ComfyUI's encoder**: `comfy/text_encoders/z_image.py`
- **Diffusers pipeline**: `diffusers/pipelines/z_image/pipeline_z_image.py`
- **Templates**: `nodes/templates/z_image_*.md`

## Related Documentation

- [Z-Image Turbo Workflow Analysis](z_image_turbo_workflow_analysis.md) - Official workflow breakdown
- [Qwen3 Official Repo](https://github.com/QwenLM/Qwen3) - Model documentation

## References

- [HuggingFace Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) - Model card
- [Qwen3 Naming Convention](https://huggingface.co/Qwen/Qwen3-4B/discussions) - No suffix = instruct model
- [Unsloth Qwen3 Docs](https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune) - Thinking mode details
