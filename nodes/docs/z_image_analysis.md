# Z-Image: ComfyUI vs Diffusers Analysis

Deep technical analysis of Z-Image text encoding differences.

**IMPORTANT: This document corrects our original analysis after testing the actual tokenizers.**

---

## Executive Summary

After testing with actual tokenizers, we discovered:

| Aspect | ComfyUI | Diffusers | Match? |
|--------|---------|-----------|--------|
| Template format | No think block | No think block | **YES** |
| Special tokens | Same IDs | Same IDs | **YES** |
| Token encoding | Identical | Identical | **YES** |
| Embedding extraction | Filtered (v2.9.11+) | Filtered by mask | **YES** |

**ComfyUI and diffusers produce IDENTICAL templates and token IDs by default.**

With `filter_padding=True` (default in v2.9.11+), embedding extraction also matches.

---

## The `enable_thinking` Confusion

The Qwen3 chat template has counterintuitive behavior:

```python
# Diffusers pipeline_z_image.py line 217-222
tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=True  # <-- Does NOT add think block!
)
```

| Parameter | Result | Meaning |
|-----------|--------|---------|
| `enable_thinking=True` | NO `<think>` block | "Let model think" (for generation) |
| `enable_thinking=False` | ADD `<think>\n\n</think>\n\n` | "Skip thinking" (pre-fill empty) |

This naming is confusing for text encoders (not generation). We renamed our parameter to `add_think_block` for clarity.

---

## Template Comparison (Verified)

### ComfyUI's Template (z_image.py:15)
```python
self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
```

### Diffusers with enable_thinking=True
```python
result = tokenizer.apply_chat_template(
    [{"role": "user", "content": "test prompt"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
# Result: "<|im_start|>user\ntest prompt<|im_end|>\n<|im_start|>assistant\n"
```

**They are IDENTICAL!**

---

## Tokenizer Comparison (Verified)

We tested both tokenizers with the same prompt:

```python
# Test results:
ComfyUI template + ComfyUI tokenizer: 10 tokens
ComfyUI template + Qwen3 tokenizer:   10 tokens
Tokens match? True

# Special tokens:
ComfyUI <|im_start|>: [151644]
Qwen3 <|im_start|>:   [151644]
```

### Why This Works

Both tokenizers have the same special token IDs for the basic format:
- `<|im_start|>` = 151644
- `<|im_end|>` = 151645
- `user`, `assistant`, etc. = same subword tokens

### Where They Differ

Tokens 151667-151668:
| Tokenizer | Token 151667 | Token 151668 |
|-----------|--------------|--------------|
| **ComfyUI (Qwen2.5-VL)** | `<|meta|>` | `<|endofmeta|>` |
| **Qwen3-4B** | `<think>` | `</think>` |

When you type `<think>` as text:
- **Qwen3-4B tokenizer**: Single token `[151667]`
- **ComfyUI tokenizer**: Subword pieces `['<th', 'ink', '>']`

---

## Embedding Extraction: The Real Difference

### Diffusers (pipeline_z_image.py:242-247)
```python
prompt_embeds = self.text_encoder(
    input_ids=text_input_ids,
    attention_mask=prompt_masks,
).hidden_states[-2]

# Filter by attention mask - only non-padded tokens
for i in range(len(prompt_embeds)):
    embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
```

### ComfyUI
Returns the full padded sequence WITH attention mask in `extra` dict:
```python
# sd1_clip.py line 293-297
extra = {}
if self.return_attention_masks:
    extra["attention_mask"] = attention_mask
return z, pooled_output, extra  # z is full padded sequence
```

The Z-Image model (lumina/model.py) uses this mask during `context_refiner` (line 535), but main transformer layers receive `mask=None` (line 542).

### Impact

| Prompt | Diffusers | ComfyUI |
|--------|-----------|---------|
| "A cat" | ~13 embeddings (no padding) | 512 embeddings + mask |
| Long prompt | ~50 embeddings (no padding) | 512 embeddings + mask |

**Key difference**: RoPE position IDs for image patches start at `cap_feats.shape[1] + 1`. With ComfyUI's longer padded sequence, image patches get different position encodings than diffusers.

This difference **cannot be fixed** without modifying ComfyUI's core CLIP system.

---

## What Our Nodes Do

### Default Behavior (add_think_block=False)

Matches diffusers exactly:
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

### Experimental (add_think_block=True)

Adds thinking block for experimentation:
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

**Note:** With ComfyUI's tokenizer, `<think>` becomes `['<th', 'ink', '>']`, not a single special token.

---

## Knobs We Expose

### ZImageTextEncoder (v2.9.5+)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `user_prompt` | - | Your prompt (renamed from `text`) |
| `template_preset` | "none" | Load from template file |
| `system_prompt` | "" | System prompt (editable after template auto-fill) |
| `add_think_block` | **False** | Add thinking tokens |
| `thinking_content` | "" | Content inside `<think>` tags |
| `assistant_content` | "" | Content after `</think>` tags |
| `raw_prompt` | "" | Bypass all formatting |

**Outputs:** conditioning, formatted_prompt, debug_output, conversation

### ZImageTurnBuilder (v2.9.5+)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `previous` | (required) | Connect from encoder or another TurnBuilder |
| `user_prompt` | - | User's message for this turn |
| `thinking_content` | "" | Assistant's thinking |
| `assistant_content` | "" | Assistant's response |
| `is_final` | True | If True, last message has no `<\|im_end\|>` |

**Outputs:** conversation, debug_output

---

## Do You Need the Qwen3-4B Tokenizer Config?

**For default behavior: No.**
The base format produces identical tokens with either tokenizer.

**For thinking experiments: Maybe.**
If you want accurate `<think>` tokenization (single token instead of subwords), you'd need:
1. Copy tokenizer_config.json from Qwen3-4B to a configs folder
2. Modify encoder to load and use it
3. This is complex and may not help

**Recommendation:**
Test with and without `add_think_block` first. If there's a noticeable difference, then consider tokenizer changes.

---

## Multi-Turn Support

**Implemented in v2.9.5** via ZImageTurnBuilder.

**Workflow:**
```
ZImageTextEncoder (first turn) → ZImageTurnBuilder (turn 2) → ZImageTextEncoder (conversation_override)
```

Each TurnBuilder adds one exchange (user message + assistant response).

**Note:** Z-Image is primarily trained on single-turn prompts. Multi-turn is experimental.

---

## Corrected Gaps Analysis

### Gap 1: Template (NO GAP)

**Original claim**: ComfyUI omits thinking tokens
**Reality**: ComfyUI matches diffusers exactly (no thinking tokens in either)
**Status**: No fix needed

### Gap 2: Sequence Length (MINOR)

**Problem**: ComfyUI allows unlimited length, diffusers uses 512
**Note**: The 512 limit in diffusers appears to be a choice, not a hard architectural limit. The DiT config (`axes_lens=[1536, 512, 512]`) shows text tokens could theoretically use up to 1536 positions. We don't know if 512 matches training data or is just conservative.
**Status**: Our nodes don't enforce 512 - users can exceed it with multi-turn conversations, which is experimental

### Gap 3: Embedding Extraction (ADDRESSED)

**Problem**: Diffusers filters to valid tokens, ComfyUI returns padded
**Solution**: Added `filter_padding` parameter (default: True) that filters embeddings by attention mask, matching DiffSynth/diffusers behavior
**Status**: Fixed in v2.9.11

### Gap 4: Thinking Token Experiments (EXPERIMENTAL)

**Hypothesis**: Adding `<think>` tokens might help embeddings
**Reality**:
- Diffusers doesn't add them
- ComfyUI's tokenizer treats them as subwords anyway
**Status**: Exposed as experimental `add_think_block` for testing

---

## Testing Methodology

### A/B Test

Use same prompt, same seed, same settings:

**Test A (Default):**
```
CLIPLoader (lumina2) -> ZImageTextEncoderSimple (add_think_block=False) -> KSampler
```

**Test B (With Think Block):**
```
CLIPLoader (lumina2) -> ZImageTextEncoderSimple (add_think_block=True) -> KSampler
```

### What to Compare

- Overall image quality
- Prompt adherence
- Fine details and textures
- Any artifacts or quality differences

---

## Recommendations

1. **Use default** (`add_think_block=False`) - matches diffusers
2. **Test experimentally** with `add_think_block=True`
3. **Don't worry about tokenizer config** for basic usage
4. **Accept embedding extraction gap** - cannot fix without core changes

---

## References

- [Diffusers Z-Image Pipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/z_image)
- [Qwen3 Official Repository](https://github.com/QwenLM/Qwen3)
- [ComfyUI z_image.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/text_encoders/z_image.py)

---

**Last Updated:** 2025-11-28
**Version:** 2.1 (updated for v2.9.5 UX redesign)
