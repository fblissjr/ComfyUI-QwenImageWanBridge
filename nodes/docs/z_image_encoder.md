# Z-Image Text Encoder

**Category:** ZImage/Encoding

## Overview

Custom encoder nodes for Z-Image that expose experimental parameters for testing. Z-Image is Alibaba's 6B parameter text-to-image model using Qwen3-4B as its text encoder.

**Key Finding:** After analysis, we found ComfyUI and diffusers produce **identical templates** by default. Our nodes match diffusers exactly, with optional experimental parameters.

**Nodes:**
- **ZImageTextEncoder** - Full-featured with templates, system prompts, multi-turn support
- **ZImageMessageChain** - Build multi-turn conversations

---

## Quick Start

The official workflow uses:
```
CLIPLoader (lumina2) -> CLIPTextEncode -> KSampler
```

Our nodes:
```
CLIPLoader (lumina2) -> ZImageTextEncoder -> KSampler
```

Default behavior matches diffusers exactly. Use templates or multi-turn for advanced control.

---

## Prerequisites

### Required Models

**Text Encoder (in `models/text_encoders/`):**
- `qwen_3_4b.safetensors` - Qwen3-4B weights

**DiT Model (in `models/diffusion_models/`):**
- `z_image_turbo_bf16.safetensors` - Z-Image Turbo model

**VAE (in `models/vae/`):**
- `ae.safetensors` - Z-Image autoencoder (16-channel)

### Model Downloads

- **Z-Image Turbo**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- **Qwen3-4B**: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

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

---

## Nodes

### ZImageTextEncoder (Full-Featured)

Full encoder with system prompts, templates, and multi-turn conversation support.

#### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| clip | CLIP | Yes | - | Z-Image CLIP model (lumina2 type) |
| text | STRING | Yes | "" | Your prompt (ignored if conversation_override connected) |
| conversation_override | ZIMAGE_CONVERSATION | No | - | Connect ZImageMessageChain output (overrides all below) |
| template_preset | ENUM | No | "none" | Template from `nodes/templates/z_image/` (auto-fills system_prompt) |
| system_prompt | STRING | No | "" | Editable system prompt (auto-filled by template via JS) |
| raw_prompt | STRING | No | "" | RAW MODE: Bypass all formatting, use your own tokens |
| add_think_block | BOOLEAN | No | **False** | Add `<think></think>` block (auto-enabled if thinking_content provided) |
| thinking_content | STRING | No | "" | Content INSIDE `<think>...</think>` tags |
| assistant_content | STRING | No | "" | Content AFTER `</think>` tags |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| conditioning | CONDITIONING | Encoded text embeddings (2560 dimensions) |
| formatted_prompt | STRING | Exact prompt that was encoded (for debugging) |

### ZImageMessageChain (Multi-Turn Conversations)

Build multi-turn conversations by chaining messages together. Connect the final output to ZImageTextEncoder's `conversation_override` input.

#### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| role | ENUM | Yes | "user" | Message role: system, user, or assistant |
| content | STRING | Yes | "" | Message content |
| previous | ZIMAGE_CONVERSATION | No | - | Previous conversation chain |
| thinking_content | STRING | No | "" | Thinking content (assistant role only, requires enable_thinking) |
| enable_thinking | BOOLEAN | No | False | Enable thinking mode (only when starting new conversation) |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| conversation | ZIMAGE_CONVERSATION | Conversation chain to pass to next node or encoder |

#### Usage

Chain multiple ZImageMessageChain nodes:

```
[ZImageMessageChain]          [ZImageMessageChain]          [ZImageMessageChain]
role: system          -->     role: user            -->     role: assistant
content: "You are..."         content: "Draw a cat"         content: ""
enable_thinking: True         previous: (connect)           thinking_content: "Thinking..."
                                                            previous: (connect)
                                                                    |
                                                                    v
                                                        [ZImageTextEncoder]
                                                        conversation_override: (connect)
```

When `enable_thinking=True` (set on the first node), all assistant messages will include `<think></think>` tags. Empty `thinking_content` produces empty tags.

---

## Workflows

### Basic T2I (matches diffusers)

```
CLIPLoader (lumina2) --> ZImageTextEncoder --> KSampler
                              |
                         add_think_block=False (default)
```

### With System Prompts

```
CLIPLoader (lumina2) --> ZImageTextEncoder --> KSampler
                              |
                         template_preset="photorealistic"
                         add_think_block=False
```

### With Raw Mode

For complete control over the prompt format:

```
raw_prompt: "<|im_start|>system
You are a helpful image generator.<|im_end|>
<|im_start|>user
A beautiful sunset over mountains<|im_end|>
<|im_start|>assistant
"
```

When `raw_prompt` is set, it bypasses all other formatting.

### Multi-Turn Conversation

```
[ZImageMessageChain]     [ZImageMessageChain]     [ZImageMessageChain]
role: system       -->   role: user         -->   role: assistant   --> ZImageTextEncoder
content: "..."           content: "prompt"        content: ""           conversation_override: (connect)
enable_thinking: True    previous: (connect)      thinking_content: ""  clip: (from CLIPLoader)
                                                  previous: (connect)
```

Build iterative conversations with context. Each ZImageMessageChain adds one message to the chain.

---

## Settings Reference

### Z-Image Turbo Sampling

| Parameter | Value | Notes |
|-----------|-------|-------|
| steps | 9 | Very low - turbo distilled model |
| cfg | 1 | No guidance - CFG baked in via DMD |
| sampler | euler | Simple, fast |
| scheduler | simple | Basic scheduling |
| denoise | 1.0 | Full denoising for T2I |

### Resolution

**Native resolution:** 1024x1024

**Supported ratios:**
- 1:1 (1024x1024) - default
- 4:3 (1024x768)
- 3:4 (768x1024)
- 16:9 (1024x576)
- 9:16 (576x1024)

### VRAM Estimates

- 1024x1024 @ 9 steps: ~8-10GB
- Higher resolutions may require more

---

## Prompting Tips

### What Works Well

**Structure your prompts:**
```
[Subject], [Setting], [Lighting], [Style], [Quality modifiers]
```

**Example:**
```
A majestic lion resting on a savanna rock, golden hour sunset,
dramatic rim lighting, wildlife photography, highly detailed, 8k
```

### Include Details

- **Lighting**: "soft diffused light", "dramatic shadows", "golden hour"
- **Style**: "photorealistic", "cinematic", "editorial", "artistic"
- **Quality**: "highly detailed", "sharp focus", "professional"
- **Mood**: "serene", "dramatic", "mysterious", "vibrant"

### Text Rendering

Z-Image can render text. Put text in quotes:
```
A neon sign that says "OPEN 24 HOURS" in a rainy alley at night
```

### Negative Prompts

Keep simple for turbo model:
```
blurry, ugly, bad quality, watermark, text, logo, distorted
```

---

## Troubleshooting

### "Model not found" errors

**Solution:** Verify model paths match the structure above.

### "Wrong CLIP type" or dimension errors

**Solution:** Ensure CLIPLoader uses `type: lumina2`

### Poor image quality

**Solutions:**
1. Use our encoder (default settings match diffusers)
2. Verify you're using our node, not CLIPTextEncode
3. Check sampling settings (steps=9, cfg=1 for turbo)

### Different results than diffusers

**Known differences:**
- Embedding extraction (we include padding, diffusers filters)
- Tokenizer bundling (minor differences)
- Sampling implementation (ComfyUI vs diffusers)

### Out of Memory (OOM)

**Solutions:**
1. Lower resolution (768x768 instead of 1024x1024)
2. Enable CPU offloading in ComfyUI settings
3. Close other applications

---

## Template Files

Templates stored in `nodes/templates/z_image/` subfolder:

| Template | Description |
|----------|-------------|
| `default` | No system prompt (matches diffusers default) |
| `photorealistic` | Photography with natural lighting |
| `bilingual_text` | English/Chinese text rendering |
| `artistic` | Creative compositions |

See the folder for additional templates.

---

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

### Formatted Prompt Examples

This section shows the exact output of `formatted_prompt` for every scenario. The prompt always ends WITHOUT `<|im_end|>` because the model is still "generating" (for embedding extraction).

---

#### ZImageTextEncoder Variations

**1. Minimal (matches diffusers default)**

Settings: `text="A cat sleeping"`, everything else default

```
<|im_start|>user
A cat sleeping<|im_end|>
<|im_start|>assistant

```

---

**2. With System Prompt**

Settings: `text="A cat sleeping"`, `system_prompt="Generate a photorealistic image."`

```
<|im_start|>system
Generate a photorealistic image.<|im_end|>
<|im_start|>user
A cat sleeping<|im_end|>
<|im_start|>assistant

```

---

**3. With Think Block (empty)**

Settings: `text="A cat sleeping"`, `add_think_block=True`

```
<|im_start|>user
A cat sleeping<|im_end|>
<|im_start|>assistant
<think>

</think>

```

---

**4. With Think Block + Thinking Content**

Settings: `text="A cat sleeping"`, `thinking_content="Soft lighting, peaceful mood, curled up position."`

(Note: `add_think_block` auto-enables when `thinking_content` is provided)

```
<|im_start|>user
A cat sleeping<|im_end|>
<|im_start|>assistant
<think>
Soft lighting, peaceful mood, curled up position.
</think>

```

---

**5. With Think Block + Thinking + Assistant Content**

Settings: `text="A cat sleeping"`, `thinking_content="Soft lighting, peaceful mood."`, `assistant_content="Creating a cozy scene..."`

```
<|im_start|>user
A cat sleeping<|im_end|>
<|im_start|>assistant
<think>
Soft lighting, peaceful mood.
</think>

Creating a cozy scene...
```

---

**6. Full Example (System + Think + Assistant)**

Settings:
- `text="A cat sleeping on a windowsill"`
- `system_prompt="You are an expert photographer."`
- `thinking_content="Golden hour light, shallow depth of field."`
- `assistant_content="Capturing the peaceful moment..."`

```
<|im_start|>system
You are an expert photographer.<|im_end|>
<|im_start|>user
A cat sleeping on a windowsill<|im_end|>
<|im_start|>assistant
<think>
Golden hour light, shallow depth of field.
</think>

Capturing the peaceful moment...
```

---

**7. Raw Prompt (Complete Control)**

Settings: `raw_prompt="<|im_start|>user\nMy custom prompt<|im_end|>\n<|im_start|>assistant\n"`

Output is exactly what you provide - no processing:

```
<|im_start|>user
My custom prompt<|im_end|>
<|im_start|>assistant

```

---

#### ZImageMessageChain Variations

The message chain builds conversations by connecting nodes. Each node adds one message.

**8. Simple User Message (no chain)**

Single ZImageMessageChain: `role="user"`, `content="A sunset over mountains"`

```
<|im_start|>user
A sunset over mountains
```

(Note: Last message has NO `<|im_end|>` - still being "generated")

---

**9. System + User Chain**

Chain: `[system] -> [user]`

Node 1: `role="system"`, `content="Generate detailed landscapes."`, `enable_thinking=False`
Node 2: `role="user"`, `content="A sunset over mountains"`, `previous=(connect)`

```
<|im_start|>system
Generate detailed landscapes.<|im_end|>
<|im_start|>user
A sunset over mountains
```

---

**10. System + User + Assistant Chain (no thinking)**

Chain: `[system] -> [user] -> [assistant]`

Node 1: `role="system"`, `content="You are an artist."`, `enable_thinking=False`
Node 2: `role="user"`, `content="Paint a sunset"`, `previous=(connect)`
Node 3: `role="assistant"`, `content="I will create a vibrant scene."`, `previous=(connect)`

```
<|im_start|>system
You are an artist.<|im_end|>
<|im_start|>user
Paint a sunset<|im_end|>
<|im_start|>assistant
I will create a vibrant scene.
```

---

**11. System + User + Assistant Chain (with thinking enabled, empty)**

Chain: `[system] -> [user] -> [assistant]`

Node 1: `role="system"`, `content="You are an artist."`, `enable_thinking=True`
Node 2: `role="user"`, `content="Paint a sunset"`, `previous=(connect)`
Node 3: `role="assistant"`, `content=""`, `thinking_content=""`, `previous=(connect)`

```
<|im_start|>system
You are an artist.<|im_end|>
<|im_start|>user
Paint a sunset<|im_end|>
<|im_start|>assistant
<think>

</think>

```

---

**12. System + User + Assistant Chain (with thinking content)**

Chain: `[system] -> [user] -> [assistant]`

Node 1: `role="system"`, `content="You are an artist."`, `enable_thinking=True`
Node 2: `role="user"`, `content="Paint a sunset"`, `previous=(connect)`
Node 3: `role="assistant"`, `content="Beginning the painting..."`, `thinking_content="Warm oranges, purples, silhouetted mountains."`, `previous=(connect)`

```
<|im_start|>system
You are an artist.<|im_end|>
<|im_start|>user
Paint a sunset<|im_end|>
<|im_start|>assistant
<think>
Warm oranges, purples, silhouetted mountains.
</think>

Beginning the painting...
```

---

**13. Multi-Turn Conversation (ending on user)**

Chain: `[system] -> [user] -> [assistant] -> [user]`

Node 1: `role="system"`, `content="You are a painter."`, `enable_thinking=True`
Node 2: `role="user"`, `content="Paint a cat"`, `previous=(connect)`
Node 3: `role="assistant"`, `content="Here is a tabby cat."`, `thinking_content="Orange fur, green eyes."`, `previous=(connect)`
Node 4: `role="user"`, `content="Now make it sleeping"`, `previous=(connect)`

```
<|im_start|>system
You are a painter.<|im_end|>
<|im_start|>user
Paint a cat<|im_end|>
<|im_start|>assistant
<think>
Orange fur, green eyes.
</think>

Here is a tabby cat.<|im_end|>
<|im_start|>user
Now make it sleeping
```

This is the most common pattern - the conversation provides context, and the final user message is the actual generation request.

---

**14. Multi-Turn with Assistant Continuation**

Chain: `[system] -> [user] -> [assistant] -> [user] -> [assistant]`

Node 1: `role="system"`, `content="You are a painter."`, `enable_thinking=True`
Node 2: `role="user"`, `content="Paint a cat"`, `previous=(connect)`
Node 3: `role="assistant"`, `content="Here is a tabby cat."`, `thinking_content="Orange fur, green eyes."`, `previous=(connect)`
Node 4: `role="user"`, `content="Make it sleeping"`, `previous=(connect)`
Node 5: `role="assistant"`, `content="Adjusting the pose..."`, `thinking_content="Curled up, eyes closed, peaceful breathing."`, `previous=(connect)`

```
<|im_start|>system
You are a painter.<|im_end|>
<|im_start|>user
Paint a cat<|im_end|>
<|im_start|>assistant
<think>
Orange fur, green eyes.
</think>

Here is a tabby cat.<|im_end|>
<|im_start|>user
Make it sleeping<|im_end|>
<|im_start|>assistant
<think>
Curled up, eyes closed, peaceful breathing.
</think>

Adjusting the pose...
```

Here the assistant starts responding - the model continues from "Adjusting the pose..."

---

#### Summary Table

| Scenario | System | Think Block | Thinking Content | Assistant Content | Ends On |
|----------|--------|-------------|------------------|-------------------|---------|
| 1. Minimal | - | - | - | - | assistant |
| 2. With system | Yes | - | - | - | assistant |
| 3. Empty think | - | Yes | - | - | assistant |
| 4. With thinking | - | Auto | Yes | - | assistant |
| 5. Full encoder | - | Auto | Yes | Yes | assistant |
| 6. Complete | Yes | Auto | Yes | Yes | assistant |
| 7. Raw | N/A | N/A | N/A | N/A | user choice |
| 8. Simple chain | - | - | - | - | user |
| 9. Sys+user | Yes | - | - | - | user |
| 10. No thinking | Yes | - | - | Yes | assistant |
| 11. Empty think | Yes | Yes | - | - | assistant |
| 12. With thinking | Yes | Yes | Yes | Yes | assistant |
| 13. Multi-turn | Yes | Yes | Yes | Yes | user |
| 14. Continuation | Yes | Yes | Yes | Yes | assistant |

**Key Rules:**
- Last message never gets `<|im_end|>` (model is "generating")
- `add_think_block` auto-enables when `thinking_content` is provided
- Empty `thinking_content` with `add_think_block=True` produces empty `<think></think>` tags
- `enable_thinking` on first ZImageMessageChain applies to ALL assistant messages in chain
- Chain messages inherit `enable_thinking` from the conversation start

**Where to End the Chain:**
- **End on user** (examples 8, 9, 13): Most natural - conversation provides context, user makes final request
- **End on assistant** (examples 10-12, 14): Provide a starting point or guide the generation direction
- For image generation, ending on user is typically preferred - the final user message is the generation prompt

---

### Qwen3 Model Variants

As of the 2507 release, there are three Qwen3 model types:

| Variant | Thinking Mode | Notes |
|---------|---------------|-------|
| **Qwen3-4B** (2504 hybrid) | Switchable | Z-Image uses this |
| Qwen3-Instruct-2507 | Never | Parameter not supported |
| Qwen3-Thinking-2507 | Always | Always thinks |

**Z-Image uses the 2504 hybrid model** (Qwen3-4B without suffix).

---

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

### Gap 2: Sequence Length (Intentional)

**Diffusers** uses `max_sequence_length=512` with truncation.

**ComfyUI** uses effectively unlimited length (`max_length=99999999`).

**Our decision**: We follow ComfyUI's unlimited approach. Qwen3-4B supports 40K tokens (`max_position_embeddings: 40960`), so truncating to 512 is unnecessarily restrictive for complex prompts.

### Gap 3: Tokenizer Special Tokens (Minor)

ComfyUI bundles a Qwen2.5-VL tokenizer where token 151667 = `<|meta|>`.
Qwen3-4B tokenizer has token 151667 = `<think>`.

When `add_think_block=True`, ComfyUI tokenizes `<think>` as subwords `['<th', 'ink', '>']` instead of a single special token.

---

## Experiments

These experiments help determine if various options improve output quality.

### Experiment 1: Think Block A/B Test

**Goal:** Does adding `<think>` block improve or change quality?

**Setup:**
1. Same prompt, same seed, same settings
2. Run A: `add_think_block=False` (matches diffusers)
3. Run B: `add_think_block=True` (experimental)
4. Compare outputs

### Experiment 2: Custom Thinking Content

**Goal:** Does providing reasoning inside `<think>` tags affect output?

**Setup:**
1. Use ZImageTextEncoder
2. Run A: Leave `thinking_content` empty
3. Run B: Provide structured reasoning in `thinking_content`:

```
thinking_content: "Key elements: [list main subjects]
Composition: [describe layout, focal points]
Lighting: [specify lighting style]
Style: [artistic direction]"
```

### Experiment 3: System Prompts

**Goal:** Do system prompts affect embedding quality?

**Setup:**
1. Same prompt across all tests
2. Test each `template_preset`:
   - `none` (diffusers default)
   - `photorealistic`
   - `artistic`
   - `bilingual_text`

---

## File Locations

- **Our encoder**: `nodes/z_image_encoder.py`
- **ComfyUI's encoder**: `comfy/text_encoders/z_image.py`
- **Diffusers pipeline**: `diffusers/pipelines/z_image/pipeline_z_image.py`
- **Templates**: `nodes/templates/z_image/` (subfolder with `.md` files)
- **JS auto-fill**: `web/js/template_autofill.js`
- **API endpoint**: `/api/z_image_templates` (registered in `__init__.py`)

## Related Documentation

- [Z-Image Turbo Workflow Analysis](z_image_turbo_workflow_analysis.md) - Official workflow breakdown

---

**Last Updated:** 2025-11-28
**Version:** 3.1 (added comprehensive formatted prompt examples)
