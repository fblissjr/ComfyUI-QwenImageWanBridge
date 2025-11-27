# Z-Image Implementation Analysis

Comprehensive analysis of Z-Image text encoding: what our nodes do, why they exist, and how ComfyUI, Diffusers, and official Qwen3 implementations compare.

---

## Executive Summary

Z-Image uses Qwen3-4B as its text encoder. ComfyUI's built-in implementation has a critical bug: it omits thinking tokens that the model was trained with. Our nodes fix this and match the diffusers implementation.

**Bottom line:** Replace `CLIPTextEncode` with `ZImageTextEncoderSimple` for better embeddings.

---

## The Problem

### What ComfyUI Does

ComfyUI hardcodes this template in `comfy/text_encoders/z_image.py`:

```python
class ZImageTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, ...):
        self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
```

This produces:
```
<|im_start|>user
A photo of a cat<|im_end|>
<|im_start|>assistant
```

### What Diffusers Does

Diffusers uses `apply_chat_template` with thinking mode enabled:

```python
# From diffusers/pipelines/z_image/pipeline_z_image.py
messages = [{"role": "user", "content": prompt_item}]
prompt_item = self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # <-- THE KEY DIFFERENCE
)
```

This produces:
```
<|im_start|>user
A photo of a cat<|im_end|>
<|im_start|>assistant
<think>

</think>

```

### Why This Matters

Qwen3-4B (no suffix) is the **instruct model**, not the base model. This is opposite of standard naming convention:
- `Qwen3-4B` = instruct model (trained with thinking)
- `Qwen3-4B-Base` = base model (no thinking)

The instruct model was trained with thinking mode support. When you omit the `<think>` tokens:
1. The token sequence doesn't match training distribution
2. Embeddings are subtly out-of-distribution
3. Image quality may degrade

---

## Three-Way Comparison

### Template Generation

| Implementation | Template | Thinking Tokens |
|----------------|----------|-----------------|
| **ComfyUI** | Hardcoded string | Missing |
| **Diffusers** | `apply_chat_template(enable_thinking=True)` | Present |
| **Our Nodes** | Manual construction matching diffusers | Present |

### Sequence Length

| Implementation | Max Length | Truncation |
|----------------|------------|------------|
| **ComfyUI** | 99999999 (unlimited) | No |
| **Diffusers** | 512 (default) | Yes |
| **Our Nodes** | 512 (configurable) | Warning only* |

*ComfyUI's tokenizer doesn't expose truncation, so we warn when prompts exceed limit.

### Embedding Extraction

| Implementation | Method | Result |
|----------------|--------|--------|
| **ComfyUI** | Returns full tensor | Includes padding embeddings |
| **Diffusers** | Filters by attention mask | Variable-length, no padding |
| **Our Nodes** | Uses ComfyUI's method | Includes padding embeddings* |

*Cannot fix without modifying ComfyUI core.

### Tokenizer Configuration

| Implementation | Tokenizer Source | Template Support |
|----------------|------------------|------------------|
| **ComfyUI** | Bundled Qwen2.5 config | No thinking template |
| **Diffusers** | HuggingFace Qwen3 | Full thinking support |
| **Our Nodes** | Uses ComfyUI's tokenizer | Manual template construction |

---

## Official Qwen3 Guidance

From the official Qwen3 repository:

### Model Variants

| Variant | Thinking Mode | Use Case |
|---------|---------------|----------|
| **Qwen3-4B** (no suffix) | Hybrid (switchable) | Z-Image uses this |
| Qwen3-4B-Base | None | Not for Z-Image |
| Qwen3-Instruct-2507 | Never | Newer, no thinking |
| Qwen3-Thinking-2507 | Always | Newer, always thinks |

### Recommended Settings

From Qwen3 README:
- **dtype**: Use `torch_dtype="auto"` (defaults to bfloat16)
- **Sampling**: Temperature 0.6-0.7, top_p 0.8-0.95, top_k 20
- **Warning**: "DO NOT use greedy decoding" (causes repetition)

### System Prompts

From Qwen3 docs:
> "Starting with Qwen3, no default system messages are used."

This means Z-Image's default (no system prompt) aligns with Qwen3's design.

---

## What Our Nodes Fix

### Gap 1: Thinking Tokens (FIXED)

**Problem**: ComfyUI omits `<think>\n\n</think>\n\n` tokens
**Solution**: Our nodes manually construct the template with thinking tokens
**Impact**: Embeddings now match diffusers training distribution

### Gap 2: Sequence Length (FIXED)

**Problem**: ComfyUI allows unlimited length, diffusers uses 512
**Solution**: `max_sequence_length` parameter (default 512) with warning
**Impact**: Prevents potential OOM and training mismatch

### Gap 3: Embedding Extraction (CANNOT FIX)

**Problem**: Diffusers filters to valid tokens only, ComfyUI returns padded sequence
**Why we can't fix**: Would require modifying `clip.encode_from_tokens_scheduled()`
**Impact**: Semantic difference in what DiT receives, likely minor

### Gap 4: Bundled Tokenizer (CANNOT FIX)

**Problem**: ComfyUI bundles Qwen2.5 tokenizer without thinking template
**Why we can't fix**: Tokenizer is bundled in ComfyUI core
**Impact**: We work around by manual template construction

---

## Expected Results

### With ComfyUI's CLIPTextEncode

- Embeddings use wrong template (missing thinking tokens)
- Token sequence doesn't match model's training distribution
- Potential quality degradation (subtle, may vary by prompt)
- Unlimited sequence length (risk of OOM or truncation issues)

### With Our ZImageTextEncoderSimple

- Embeddings use correct template (with thinking tokens)
- Token sequence matches diffusers implementation
- Expected quality improvement (especially for complex prompts)
- Default 512 token limit with warnings

### With Our ZImageTextEncoder (Full)

All benefits of Simple, plus:
- System prompt presets (quality, photorealistic, artistic, bilingual)
- Custom system prompt support
- Template file support (`nodes/templates/z_image_*.md`)
- Debug mode showing formatted prompt

---

## Testing Methodology

To verify the thinking token fix improves output:

### A/B Test Setup

1. **Same prompt**: Use identical text for both tests
2. **Same seed**: Use fixed seed for reproducibility
3. **Same settings**: 9 steps, CFG=1, euler sampler

### Test A: ComfyUI Default

```
CLIPLoader (lumina2) -> CLIPTextEncode -> KSampler
```

### Test B: Our Fix

```
CLIPLoader (lumina2) -> ZImageTextEncoderSimple (enable_thinking=True) -> KSampler
```

### What to Compare

- Overall image quality and coherence
- Prompt adherence (does it match what you asked for?)
- Fine details and textures
- Text rendering (if prompt includes text)

---

## Technical Deep Dive

### Embedding Dimensions

| Component | Dimension |
|-----------|-----------|
| Qwen3-4B hidden size | 2560 |
| Z-Image DiT cap_feat_dim | 2560 |
| DiT internal dimension | 3840 |
| Projection | Linear(2560 -> 3840) with RMSNorm |

### Hidden State Extraction

Both ComfyUI and Diffusers use `hidden_states[-2]` (second-to-last layer):

```python
# Diffusers
prompt_embeds = text_encoder(...).hidden_states[-2]

# ComfyUI (via layer_idx=-2)
intermediate = x.clone()  # at layer -2
```

### Token IDs (Qwen3)

| Token | ID | Purpose |
|-------|-----|---------|
| `<\|endoftext\|>` | 151643 | End of document / padding |
| `<\|im_end\|>` | 151645 | End of turn (eos) |
| `</think>` | 151668 | Thinking content delimiter |

---

## Recommendations

### For Most Users

Use `ZImageTextEncoderSimple`:
- Drop-in replacement for CLIPTextEncode
- Adds thinking tokens automatically
- Minimal overhead
- Default settings match diffusers

### For Experimentation

Use `ZImageTextEncoder`:
- Try system prompts (photorealistic, artistic, etc.)
- Use debug mode to inspect formatted prompts
- Adjust max_sequence_length if needed
- Create custom templates in `nodes/templates/`

### For Debugging

Enable `debug_mode=True` to see:
- Full formatted prompt with thinking tokens
- Estimated vs actual sequence length
- System prompt being used
- Conditioning tensor shape

---

## References

- [Diffusers Z-Image Pipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/z_image)
- [Qwen3 Official Repository](https://github.com/QwenLM/Qwen3)
- [HuggingFace Qwen3-4B Model Card](https://huggingface.co/Qwen/Qwen3-4B)
- [Unsloth Qwen3 Documentation](https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune)
- [ComfyUI z_image.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/text_encoders/z_image.py)

---

**Last Updated:** 2025-11-27
