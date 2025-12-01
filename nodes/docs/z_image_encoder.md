# Z-Image Text Encoder

**Category:** ZImage/Encoding

> **New here?** Start with [z_image_intro.md](z_image_intro.md) for a quick overview of what these nodes do and why you might use them.

## Overview

Custom encoder nodes for Z-Image that expose experimental parameters for testing. Z-Image is Alibaba's 6B parameter text-to-image model using Qwen3-4B as its text encoder.

**Key Finding:** After analysis, we found ComfyUI and diffusers produce **identical templates** by default. Our nodes match diffusers exactly, with optional experimental parameters.

**Nodes:**
- **ZImageTextEncoder** - Full-featured with templates, system prompts, multi-turn support (outputs conversation for chaining)
- **ZImageTextEncoderSimple** - Simplified encoder for quick use / negative prompts (no conversation chaining)
- **ZImageTurnBuilder** - Add conversation turns for multi-turn workflows (user+assistant per turn)
- **PromptKeyFilter** - Strip quotes from JSON keys to prevent them appearing as text

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
- `ae.safetensors` - Z-Image autoencoder (Flux-derived, 16-channel)

> **Note on VAE:** Z-Image uses a Flux-derived VAE, not a Wan VAE. While other 16-channel VAEs share the same tensor shape, they have different scaling factors. Using non-official VAEs (like Wan2.1-upscale2x) is experimental and may cause color shifts or quality differences.

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

### Reference Implementation Note

**DiffSynth-Studio always uses thinking tags** with empty content by default:
```python
# From DiffSynth z_image.py
pipe.tokenizer.apply_chat_template(
    messages,
    enable_thinking=True,  # Always enabled
)
```

This produces empty `<think></think>` tags in every prompt. Our `add_think_block=True` default matches this behavior. If you want to match the bare ComfyUI/diffusers behavior (no think block), set `add_think_block=False`.

### Embedding Extraction: Padding Filter

We verified how different implementations handle text embeddings:

| Implementation | Filters Padding? |
|----------------|------------------|
| **diffusers** (HuggingFace) | Yes |
| **DiffSynth** | Yes |
| **ComfyUI** (stock) | No |

Both reference implementations filter padding tokens before sending to the DiT:

```python
# From diffusers pipeline_z_image.py (lines 242-247)
embeddings_list = []
for i in range(len(prompt_embeds)):
    embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
```

### `filter_padding` Parameter

Our nodes include a `filter_padding` parameter (default: `True`) that matches the diffusers/DiffSynth behavior.

| filter_padding | Behavior |
|----------------|----------|
| `True` (default) | Matches diffusers/DiffSynth - filters padding tokens |
| `False` | Stock ComfyUI behavior - sends padded sequence + mask |

**Why this matters:** Short prompts have high padding ratios (e.g., 50 tokens + 462 padding = 91% padding). While the DiT can handle padding via its learned `cap_pad_token`, the reference implementations chose to filter it out.

**When to disable:** Set `filter_padding=False` if you want to match stock ComfyUI behavior or debug embedding-related issues.

---

## Nodes

### ZImageTextEncoder (Full-Featured)

Full encoder with system prompts, templates, and multi-turn conversation support. Handles a complete first turn (system + user + assistant) and outputs conversation for chaining.

#### Inputs (in order)

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| clip | CLIP | Yes | - | Z-Image CLIP model (lumina2 type) |
| user_prompt | STRING | Yes | "" | Your prompt - what you want the model to generate |
| conversation_override | ZIMAGE_CONVERSATION | No | - | Connect from ZImageTurnBuilder (uses this instead of building one) |
| template_preset | ENUM | No | "none" | Template from `nodes/templates/z_image/` (auto-fills system_prompt) |
| system_prompt | STRING | No | "" | Editable system prompt (auto-filled by template via JS) |
| add_think_block | BOOLEAN | No | **True** | Add `<think></think>` block (matches DiffSynth/diffusers reference implementations) |
| thinking_content | STRING | No | "" | Content INSIDE `<think>...</think>` tags |
| assistant_content | STRING | No | "" | Content AFTER `</think>` tags |
| raw_prompt | STRING | No | "" | RAW MODE: Bypass ALL formatting, use your own tokens |
| strip_key_quotes | BOOLEAN | No | **False** | Remove quotes from JSON keys (e.g., `"subject":` becomes `subject:`) |
| filter_padding | BOOLEAN | No | **True** | Filter padding tokens (matches diffusers/DiffSynth) |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| conditioning | CONDITIONING | Encoded text embeddings (2560 dimensions) |
| formatted_prompt | STRING | Exact prompt that was encoded (for debugging) |
| debug_output | STRING | Detailed breakdown: mode, char counts, token estimate |
| conversation | ZIMAGE_CONVERSATION | Chain to ZImageTurnBuilder for multi-turn |

### ZImageTextEncoderSimple (Quick Encoding)

Simplified encoder for quick use - ideal for **negative prompts**. Same template/thinking support as the full encoder, but without conversation chaining.

#### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| clip | CLIP | Yes | - | Z-Image CLIP model (lumina2 type) |
| user_prompt | STRING | Yes | "" | Your prompt (what you want or don't want) |
| template_preset | ENUM | No | "none" | Template from `nodes/templates/z_image/` |
| system_prompt | STRING | No | "" | System instructions |
| add_think_block | BOOLEAN | No | **True** | Add `<think></think>` block (matches reference implementations) |
| thinking_content | STRING | No | "" | Content inside think tags |
| assistant_content | STRING | No | "" | Content after think tags |
| filter_padding | BOOLEAN | No | **True** | Filter padding tokens (matches diffusers/DiffSynth) |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| conditioning | CONDITIONING | Encoded text embeddings |
| formatted_prompt | STRING | Exact prompt that was encoded |
| debug_output | STRING | Mode, char counts, formatted prompt (code block) |

#### Usage

**For negative prompts:**
```
Positive: ZImageTextEncoder -> KSampler (positive)
Negative: ZImageTextEncoderSimple -> KSampler (negative)
         user_prompt: "bad anatomy, blurry, watermark, text"
```

Both use the same Qwen3-4B chat template format for consistency.

### ZImageTurnBuilder (Multi-Turn Conversations)

Add conversation turns for multi-turn workflows. Each node represents one complete turn (user message + optional assistant response).

**Two workflow options:**
1. **Without clip**: Chain to more TurnBuilders or back to encoder's `conversation_override`
2. **With clip**: Outputs conditioning directly - no need to chain back to encoder

#### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| previous | ZIMAGE_CONVERSATION | Yes | - | Previous conversation (from encoder or another TurnBuilder) |
| user_prompt | STRING | Yes | "" | User's message for this turn |
| clip | CLIP | No | - | Connect to encode directly (skip chaining back to encoder) |
| thinking_content | STRING | No | "" | Assistant's thinking (only if conversation has enable_thinking=True) |
| assistant_content | STRING | No | "" | Assistant's response after thinking |
| is_final | BOOLEAN | No | True | Is this the last turn? If True, last message has no `<|im_end|>` |
| strip_key_quotes | BOOLEAN | No | **False** | Remove quotes from JSON keys (e.g., `"subject":` becomes `subject:`) |
| filter_padding | BOOLEAN | No | **True** | Filter padding tokens (only when clip connected) |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| conversation | ZIMAGE_CONVERSATION | Updated conversation (for chaining to more turns) |
| conditioning | CONDITIONING | Encoded embeddings (only populated if clip connected) |
| formatted_prompt | STRING | Full formatted text of conversation |
| debug_output | STRING | Turn details, char counts, and this turn's formatted messages |

#### Usage

**Option 1: Direct encoding (with clip)**

Connect clip to TurnBuilder for final turn - outputs conditioning directly:

```
[ZImageTextEncoder]              [ZImageTurnBuilder]              [KSampler]
user_prompt: "Paint a cat"  -->  user_prompt: "Make it sleep"  -->  (conditioning)
add_think_block: True            clip: (from CLIPLoader)
(outputs conversation)           is_final: True
                                 previous: (from encoder)
```

**Option 2: Chain back to encoder (without clip)**

Traditional workflow - chain conversation back to encoder:

```
[ZImageTextEncoder]              [ZImageTurnBuilder]              [ZImageTextEncoder]
user_prompt: "Paint a cat"  -->  user_prompt: "Make it sleep"  -->  conversation_override: (connect)
add_think_block: True            (no clip)                           clip: (from CLIPLoader)
(outputs conversation)           is_final: True
                                 previous: (from encoder)
```

**Key behaviors:**
- Each turn adds user + optional assistant messages
- `enable_thinking` is inherited from the encoder's `add_think_block` setting
- If `is_final=True` (default), the last message has no `<|im_end|>` (model continues)
- If `is_final=False`, all messages get `<|im_end|>` (more turns expected)
- When clip is connected, outputs conditioning directly (no need for second encoder)

---

## Workflows

### Basic T2I (matches DiffSynth reference)

```
CLIPLoader (lumina2) --> ZImageTextEncoder --> KSampler
                              |
                         user_prompt: "A cat sleeping"
                         add_think_block=True (default)
```

### With System Prompts

```
CLIPLoader (lumina2) --> ZImageTextEncoder --> KSampler
                              |
                         user_prompt: "A cat sleeping"
                         template_preset="photorealistic"
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

**Option 1: Direct encoding (recommended for single turn addition)**

```
[ZImageTextEncoder]              [ZImageTurnBuilder]              [KSampler]
user_prompt: "Paint a cat"  -->  user_prompt: "Make it sleep"  -->  (conditioning output)
system_prompt: "You are..."      clip: (from CLIPLoader)
add_think_block: True            thinking_content: "Curled up"
(outputs conversation)           assistant_content: "Adjusting"
                                 previous: (from encoder)
                                 is_final: True
```

**Option 2: Chain back to encoder (for more flexibility)**

```
[ZImageTextEncoder]              [ZImageTurnBuilder]              [ZImageTextEncoder]
user_prompt: "Paint a cat"  -->  user_prompt: "Make it sleep"  -->  conversation_override: (connect)
system_prompt: "You are..."      thinking_content: "Curled up"       clip: (from CLIPLoader)
add_think_block: True            assistant_content: "Adjusting"
(outputs conversation)           previous: (from encoder)
                                 is_final: True
```

The first encoder creates the initial conversation (system + user + assistant). Turn builders add subsequent exchanges. With clip connected to TurnBuilder, get conditioning directly. Without clip, chain to another encoder.

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

### About These Settings (Research Notes)

Based on our analysis of the Decoupled-DMD paper (arXiv:2511.22677) and the official Z-Image-Turbo model files, here's what we found:

- The official diffusers example uses `guidance_scale=0.0` with a comment "Guidance should be 0 for the Turbo models"
- The paper describes a training process where CFG patterns get embedded into the model weights during distillation
- `num_inference_steps=9` in diffusers appears to produce 8 actual DiT forwards (8 NFEs), matching the distillation target
- The scheduler is `FlowMatchEulerDiscreteScheduler` with `shift=3.0`

For more details, see `internal/z_image_paper_analysis/decoupled_dmd_training_report.md`.

### Resolution

**Native resolution:** 1024x1024

**Supported ratios:**
- 1:1 (1024x1024) - default
- 4:3 (1024x768)
- 3:4 (768x1024)
- 16:9 (1024x576)
- 9:16 (576x1024)

**Alignment:** 16 pixels (height and width must be divisible by 16)

### Token Limits

**Reference implementation limit:** 512 tokens (appears to be a choice, not a hard architectural limit)

DiffSynth and diffusers enforce `max_sequence_length=512` via truncation. However, analysis of the DiT config suggests this may not be architecturally required:

**DiT 3D RoPE Configuration:**
```
axes_lens=[1536, 512, 512]  # Max positions per axis
```

- **Axis 0 (1536)**: Sequential positions for text tokens
- **Axis 1 (512)**: Height positions for image patches
- **Axis 2 (512)**: Width positions for image patches

Text and image patches use different axes, so they don't compete for positions. The theoretical text limit is 1536 tokens on axis 0.

**Why 512 in reference implementations?**
We don't know for certain. Possibilities:
1. It may match what was used during training
2. It may just be a conservative choice
3. The low `rope_theta=256.0` (vs LLM's 1M) means position encoding is very "sharp" - going beyond positions seen during training risks artifacts

**Our nodes:** We don't enforce 512 tokens. With system prompts + thinking blocks + multi-turn conversations, you can exceed this. This is experimental - it might work fine, or you might see artifacts. If you have issues with very long prompts, try simplifying.

**Token counting (v2.9.12):** Debug output now shows actual token count:
```
=== Token Count ===
Tokens: 47 / 512 reference limit
```

If you exceed 512 tokens, you'll see a warning:
```
=== Token Count ===
Tokens: 623 / 512 reference limit
WARNING: Exceeds reference limit by 111 tokens
  diffusers/DiffSynth truncate at 512. ComfyUI allows longer sequences.
  Results may differ from reference implementations.
```

**Diagnostic:** Check `debug_output` for token count. If your prompts exceed 512 tokens and you see artifacts, try simplifying.

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

### JSON key names appearing as text in image

**Problem:** When using JSON-style prompts like `{"subject": "a cat", "style": "photo"}`, the quoted key names ("subject", "style") appear as visible text in the generated image.

**Solution:** Enable `strip_key_quotes` on the encoder (or use PromptKeyFilter node).

This converts `"subject": "description"` to `subject: "description"` - removing quotes from keys only while preserving quoted values.

---

## Utility Nodes

### PromptKeyFilter

Standalone text filter node in `QwenImage/Utilities`. Use when you need filtering separate from encoding, or with other encoders (Qwen-Image-Edit, HunyuanVideo).

**Inputs:**
| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| text | STRING | Yes | "" | Paste or type text directly |
| strip_key_quotes | BOOLEAN | Yes | **True** | Remove quotes from JSON keys |
| text_input | STRING | No | - | Connect from another node (takes priority over text field) |

**Output:** `text` (STRING) - filtered text

**Two ways to use:**
1. Paste/type directly into the `text` field
2. Connect from another node (LLM output, etc.) via `text_input`

### ZImageEmptyLatent

Creates 16-channel latents for Z-Image with **auto-alignment** to 16px. Input any dimensions - the node rounds to the nearest valid resolution.

**Inputs:**
| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| width | INT | Yes | 1024 | Width in pixels (auto-aligned to 16px) |
| height | INT | Yes | 1024 | Height in pixels (auto-aligned to 16px) |
| batch_size | INT | Yes | 1 | Number of latents to generate |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| latent | LATENT | 16-channel latent tensor |
| width | INT | Aligned width (use for downstream nodes) |
| height | INT | Aligned height (use for downstream nodes) |
| resolution_info | STRING | Human-readable alignment info |

**Examples:**
| Input | Output | Info String |
|-------|--------|-------------|
| 1211x1024 | 1216x1024 | "1211x1024 -> 1216x1024 (auto-aligned)" |
| 1000x1000 | 1008x1008 | "1000x1000 -> 1008x1008 (auto-aligned)" |
| 1024x768 | 1024x768 | "1024x768 (already aligned)" |

### LLMOutputParser

Parses LLM output (JSON/YAML/text) and exposes fields for Z-Image encoder. Works with any LLM node.

**Inputs:**
| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| text_input | STRING | Yes | - | LLM output to parse (connect from any LLM node) |
| parse_mode | ENUM | Yes | "auto" | auto, passthrough, json, yaml |
| user_prompt_key | STRING | No | "user_prompt" | JSON/YAML key for user_prompt |
| system_prompt_key | STRING | No | "system_prompt" | JSON/YAML key for system_prompt |
| thinking_key | STRING | No | "thinking" | JSON/YAML key for thinking_content |
| assistant_key | STRING | No | "assistant" | JSON/YAML key for assistant_content |
| fallback_to_passthrough | BOOLEAN | No | True | Use raw text if parsing fails |
| strip_quotes | BOOLEAN | No | False | Remove quotes from extracted fields |
| previous_conversation | ZIMAGE_CONVERSATION | No | - | Build multi-turn from previous context |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| user_prompt | STRING | Extracted user prompt |
| system_prompt | STRING | Extracted system prompt |
| thinking_content | STRING | Extracted thinking content |
| assistant_content | STRING | Extracted assistant content |
| raw_text | STRING | Original input text |
| parse_status | STRING | Status message (success/fallback/error) |
| conversation | ZIMAGE_CONVERSATION | For multi-turn chaining |

**Features:**
- Strips markdown code fences (```json, ```yaml)
- Unwraps nested structures (result.data, etc.)
- Dot notation for nested keys (e.g., "result.prompt")
- Falls back to common alternatives (prompt, text, content, message)

---

## Template Files

Templates stored in `nodes/templates/z_image/` subfolder.

### Extended Template Format (v2.9.10+)

Templates can now include thinking content in YAML frontmatter:

```yaml
---
name: z_image_json_structured
description: Parse JSON-structured image descriptions
model: z-image
category: structured
add_think_block: true
thinking_content: |
  Parsing the JSON structure to identify:
  - Subject and scene elements
  - Style and artistic direction
assistant_content: ""
---
System prompt body text here...
```

**Available frontmatter fields:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | - | Template identifier |
| `description` | string | - | Short description |
| `model` | string | - | Target model (z-image) |
| `category` | string | - | Template category |
| `add_think_block` | boolean | false | Auto-enable thinking checkbox |
| `thinking_content` | string | "" | Pre-fill thinking content field |
| `assistant_content` | string | "" | Pre-fill assistant content field |

When a template is selected, JS auto-fills all configured fields. User can edit any value before generation.

### Template Categories

**Style Templates:**
| Template | Description |
|----------|-------------|
| `default` | No system prompt (matches diffusers default) |
| `photorealistic` | Photography with natural lighting |
| `bilingual_text` | English/Chinese text rendering |
| `artistic` | Creative compositions |

**Structured Prompt Templates:**
| Template | Description |
|----------|-------------|
| `json_structured` | Parse JSON-formatted prompts (includes thinking) |
| `yaml_structured` | Parse YAML hierarchical prompts (includes thinking) |
| `markdown_structured` | Parse Markdown-formatted prompts (includes thinking) |

See the folder for 140+ additional templates.

---

## Technical Details

### Qwen3-4B Text Encoder

| Parameter | Value |
|-----------|-------|
| Hidden Size | 2560 |
| Layers | 36 |
| Vocabulary | 151,936 |
| Max Position Embeddings | 40,960 |
| Embedding Layer Used | `hidden_states[-2]` (second-to-last) |
| Attention | GQA (32 heads, 8 KV heads) |
| RoPE theta | 1,000,000 |
| dtype | bfloat16 |

### DiT (ZImageTransformer2DModel)

| Parameter | Value | Notes |
|-----------|-------|-------|
| dim | 3840 | Hidden dimension |
| n_layers | 30 | Main transformer blocks (with timestep modulation) |
| n_refiner_layers | 2 | Context refiner (no modulation) + noise refiner |
| n_heads / n_kv_heads | 30 / 30 | Full attention (not GQA like text encoder) |
| cap_feat_dim | 2560 | Caption feature dim (matches Qwen3-4B) |
| in_channels | 16 | VAE latent channels |
| all_patch_size | [2] | Spatial patch size (2x2 in latent space) |
| axes_dims | [32, 48, 48] | RoPE dimensions per axis (sum = 128 = head_dim) |
| axes_lens | [1536, 512, 512] | Max positions (sequence, height, width) |
| rope_theta | 256.0 | Much lower than text encoder's 1M |
| t_scale | 1000.0 | Timestep scaling |
| qk_norm | true | RMSNorm on Q/K for stability |

**Why different attention?** The text encoder uses GQA (4 heads share each KV pair) for memory efficiency with long sequences (up to 40k tokens). The DiT uses full attention for maximum expressiveness on shorter, fixed-length sequences (~5k tokens max).

### Formatted Prompt Examples

This section shows the exact output of `formatted_prompt` for every scenario.

**Closing tag behavior:**
- Empty `assistant_content`: No closing `<|im_end|>` (matches diffusers, model is "generating")
- With `assistant_content`: Closes with `<|im_end|>` (complete message)

---

#### ZImageTextEncoder Variations

**1. Minimal (matches diffusers default)**

Settings: `user_prompt="A cat sleeping"`, everything else default

```
<|im_start|>user
A cat sleeping<|im_end|>
<|im_start|>assistant

```

---

**2. With System Prompt**

Settings: `user_prompt="A cat sleeping"`, `system_prompt="Generate a photorealistic image."`

```
<|im_start|>system
Generate a photorealistic image.<|im_end|>
<|im_start|>user
A cat sleeping<|im_end|>
<|im_start|>assistant

```

---

**3. With Think Block (empty)**

Settings: `user_prompt="A cat sleeping"`, `add_think_block=True`

```
<|im_start|>user
A cat sleeping<|im_end|>
<|im_start|>assistant
<think>

</think>

```

---

**4. With Think Block + Thinking Content**

Settings: `user_prompt="A cat sleeping"`, `thinking_content="Soft lighting, peaceful mood, curled up position."`

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

Settings: `user_prompt="A cat sleeping"`, `thinking_content="Soft lighting, peaceful mood."`, `assistant_content="Creating a cozy scene..."`

```
<|im_start|>user
A cat sleeping<|im_end|>
<|im_start|>assistant
<think>
Soft lighting, peaceful mood.
</think>

Creating a cozy scene...<|im_end|>
```

Note: `<|im_end|>` is added because `assistant_content` is provided.

---

**6. Full Example (System + Think + Assistant)**

Settings:
- `user_prompt="A cat sleeping on a windowsill"`
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

Capturing the peaceful moment...<|im_end|>
```

Note: `<|im_end|>` is added because `assistant_content` is provided.

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

#### ZImageTurnBuilder Variations

The turn builder extends conversations created by the encoder. Each turn adds a user message and optional assistant response.

**8. Single Turn (user only, ends on user)**

Flow: `ZImageTextEncoder -> ZImageTurnBuilder -> ZImageTextEncoder`

Encoder: `user_prompt="Paint a cat"`, `add_think_block=True`, `thinking_content="Orange fur."`, `assistant_content="Here's a cat."`
TurnBuilder: `user_prompt="Make it sleeping"`, `is_final=True` (no assistant content)

```
<|im_start|>user
Paint a cat<|im_end|>
<|im_start|>assistant
<think>
Orange fur.
</think>

Here's a cat.<|im_end|>
<|im_start|>user
Make it sleeping
```

---

**9. Single Turn with Assistant Response**

Flow: `ZImageTextEncoder -> ZImageTurnBuilder -> ZImageTextEncoder`

Encoder: `user_prompt="Paint a cat"`, `add_think_block=True`, `thinking_content="Orange fur."`, `assistant_content="Here's a cat."`
TurnBuilder: `user_prompt="Make it sleeping"`, `thinking_content="Curled up, peaceful."`, `assistant_content="Adjusting..."`, `is_final=True`

```
<|im_start|>user
Paint a cat<|im_end|>
<|im_start|>assistant
<think>
Orange fur.
</think>

Here's a cat.<|im_end|>
<|im_start|>user
Make it sleeping<|im_end|>
<|im_start|>assistant
<think>
Curled up, peaceful.
</think>

Adjusting...<|im_end|>
```

Note: Both assistant messages have `<|im_end|>` because `assistant_content` is provided.

---

**10. Two-Turn Chain**

Flow: `ZImageTextEncoder -> ZImageTurnBuilder -> ZImageTurnBuilder -> ZImageTextEncoder`

Encoder: `user_prompt="Draw a house"`, `add_think_block=True`
TurnBuilder 1: `user_prompt="Add a garden"`, `assistant_content="Adding flowers."`, `is_final=False`
TurnBuilder 2: `user_prompt="Make it sunset"`, `thinking_content="Warm colors."`, `is_final=True`

```
<|im_start|>user
Draw a house<|im_end|>
<|im_start|>assistant
<think>

</think>

<|im_start|>user
Add a garden<|im_end|>
<|im_start|>assistant
<think>

</think>

Adding flowers.<|im_end|>
<|im_start|>user
Make it sunset<|im_end|>
<|im_start|>assistant
<think>
Warm colors.
</think>

```

Note: First encoder has empty `assistant_content` so no `<|im_end|>`. TurnBuilder 1 has `assistant_content` so gets `<|im_end|>`. TurnBuilder 2 is final with empty `assistant_content` so stays open.

---

**11. With System Prompt (Full Example)**

Flow: `ZImageTextEncoder -> ZImageTurnBuilder -> ZImageTextEncoder`

Encoder:
- `system_prompt="You are a painter."`
- `user_prompt="Paint a cat"`
- `add_think_block=True`
- `thinking_content="Orange fur, green eyes."`
- `assistant_content="Here is a tabby cat."`

TurnBuilder:
- `user_prompt="Now make it sleeping"`
- `thinking_content="Curled up, eyes closed."`
- `assistant_content="Adjusting the pose..."`
- `is_final=True`

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
Now make it sleeping<|im_end|>
<|im_start|>assistant
<think>
Curled up, eyes closed.
</think>

Adjusting the pose...<|im_end|>
```

Note: Both assistant messages have `<|im_end|>` because `assistant_content` is provided.

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
| 8. Turn (user only) | - | Yes | Yes | Yes | user |
| 9. Turn (with asst) | - | Yes | Yes | Yes | assistant |
| 10. Two turns | - | Yes | Yes | Yes | assistant |
| 11. Full multi-turn | Yes | Yes | Yes | Yes | assistant |

**Key Rules:**
- Last message never gets `<|im_end|>` when `is_final=True` (model is "generating")
- When `is_final=False`, all messages get `<|im_end|>` (more turns expected)
- `add_think_block` on encoder enables thinking for ALL assistant messages
- Turn builders inherit `enable_thinking` from the encoder
- Each turn builder adds 1 user message + 0-1 assistant messages

**Design Notes:**
- Most users only need the encoder (handles system + user + assistant in one node)
- Turn builders are for iterative refinement workflows
- Ending on user (no assistant content) is natural for "do this next" instructions
- Ending on assistant guides the generation direction

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

**Last Updated:** 2025-12-01
**Version:** 4.9 (v2.9.12: added ZImageEmptyLatent, LLMOutputParser docs)
