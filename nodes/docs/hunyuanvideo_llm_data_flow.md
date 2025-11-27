# HunyuanVideo 1.5 LLM Data Flow: Technical Deep Dive

This document explains exactly how the Qwen2.5-VL and byT5 language models work in HunyuanVideo 1.5, from prompt input to final conditioning, including attention mechanisms and the critical token cropping behavior.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [The Dual Encoder System](#the-dual-encoder-system)
3. [Tokenization Flow](#tokenization-flow)
4. [Template Formatting](#template-formatting)
5. [Attention and the "Influence" Mechanism](#attention-and-the-influence-mechanism)
6. [Token Cropping: What, When, Why, How](#token-cropping-what-when-why-how)
7. [Qwen2.5-VL Architecture Details](#qwen25-vl-architecture-details)
8. [byT5 Architecture and Quote Extraction](#byt5-architecture-and-quote-extraction)
9. [Vision Token Processing](#vision-token-processing)
10. [Final Conditioning Construction](#final-conditioning-construction)
11. [Practical Implications](#practical-implications)

---

## Architecture Overview

HunyuanVideo 1.5 uses a **dual text encoder** system:

```
User Prompt ──┬──> Qwen2.5-VL (7B) ──> Primary embeddings (3584-dim)
              │
              └──> byT5-small ────────> Secondary embeddings (1472-dim)
                   (quoted text only)
```

Both encoders feed into the DiT (Diffusion Transformer) for video generation. The Qwen embeddings provide semantic understanding, while byT5 handles precise text rendering (signs, labels, etc.).

---

## The Dual Encoder System

### Primary: Qwen2.5-VL-7B-Instruct

| Parameter | Value |
|-----------|-------|
| Hidden size | 3584 |
| Layers | 28 |
| Attention heads | 28 |
| KV heads | 4 (GQA) |
| Head dimension | 128 |
| Intermediate size | 18944 |
| Position encoding | M-RoPE (Multimodal Rotary) |
| Special tokens | `<\|im_start\|>` (151644), `<\|im_end\|>` (151645), etc. |

**Role:** Semantic understanding, scene composition, temporal reasoning, subject descriptions.

### Secondary: byT5-small (Glyph-SDXL-v2)

| Parameter | Value |
|-----------|-------|
| Hidden size | 1472 |
| Architecture | T5 encoder-only |
| Input | Byte-level tokens (UTF-8) |
| Attention | Relative position bias (bidirectional) |

**Role:** Precise text rendering for quoted strings ("Hello World", signs, labels).

---

## Tokenization Flow

### Step 1: Input Reception
```python
# Our node receives:
text = "A woman walks down the street"
system_prompt = "You are a helpful assistant..."
```

### Step 2: Template Formatting
```python
# HunyuanVideo15Tokenizer formats as:
formatted = """<|im_start|>system
You are a helpful assistant. Describe the video by detailing the following aspects:
1. The main content and theme of the video.
2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.
3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.
4. background environment, light, style and atmosphere.
5. camera angles, movements, and transitions used in the video.<|im_end|>
<|im_start|>user
A woman walks down the street<|im_end|>
<|im_start|>assistant
"""
```

### Step 3: Dual Tokenization

**Qwen2.5-VL tokenization:**
```python
tokens["qwen25_7b"] = [
    [(151644, 1.0),   # <|im_start|>
     (8948, 1.0),     # system
     (198, 1.0),      # \n
     # ... system prompt tokens (~60-80 tokens) ...
     (151645, 1.0),   # <|im_end|>
     (198, 1.0),      # \n
     (151644, 1.0),   # <|im_start|>
     (872, 1.0),      # user
     (198, 1.0),      # \n
     # ... user prompt tokens ...
     (151645, 1.0),   # <|im_end|>
     (198, 1.0),      # \n
     (151644, 1.0),   # <|im_start|>
     (77091, 1.0),    # assistant
     (198, 1.0)]      # \n
]
```

**byT5 tokenization (if quotes detected):**
```python
# For prompt: 'A sign saying "Hello World"'
# Extracted: ["Hello World"]
# Formatted: 'Text "Hello World". '
tokens["byt5"] = byte_level_tokens(formatted_quote_text)
```

---

## Template Formatting

### Skip Template Detection

ComfyUI checks if the input already has chat markers:
```python
# In QwenImageTokenizer.tokenize_with_weights():
skip_template = False
if text.startswith('<|im_start|>'):
    skip_template = True
if text.startswith('<|start_header_id|>'):
    skip_template = True
```

This is critical: **If you provide your own chat template, ComfyUI won't double-wrap it.**

### Our Node's Approach

```python
# In HunyuanVideoTextEncoder._encode_single():
if uses_custom_template and system_prompt:
    formatted_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
else:
    formatted_text = text  # Let ComfyUI apply default template
```

When we provide a system prompt, we format it ourselves. ComfyUI sees `<|im_start|>` at the start and skips its default template.

---

## Attention and the "Influence" Mechanism

This is the key insight: **System prompt tokens influence user embeddings through causal attention.**

### How It Works

```
Token sequence: [sys_tokens...] [user_tokens...] [assistant_marker]
                     ↓                ↓
Attention:     sys attends to    user attends to
               itself only       sys + itself
```

During the forward pass through Qwen2.5-VL's 28 transformer layers:

1. **System tokens** (positions 0 to ~60) can only attend to themselves (causal mask)
2. **User tokens** (positions ~61 to N) can attend to **all system tokens + previous user tokens**
3. **Assistant token** can attend to everything before it

### The Attention Computation

```python
# In Attention.forward():
# For each layer:
xq = self.q_proj(hidden_states)  # Query
xk = self.k_proj(hidden_states)  # Key
xv = self.v_proj(hidden_states)  # Value

# Causal mask: position i can only attend to positions 0..i
causal_mask = torch.empty(seq_len, seq_len).fill_(float("-inf")).triu_(1)
#     [0, -inf, -inf, -inf]
#     [0,    0, -inf, -inf]
#     [0,    0,    0, -inf]
#     [0,    0,    0,    0]

# Attention: softmax(QK^T / sqrt(d)) * V
output = optimized_attention(xq, xk, xv, heads, mask=causal_mask)
```

### Why This Matters

When computing the embedding for user token "woman":
```
attention_weights = softmax([
    score(woman, system_token_1),
    score(woman, system_token_2),
    ...
    score(woman, "Describe"),
    score(woman, "video"),
    ...
    score(woman, "A"),       # first user token
    score(woman, "woman")    # self
])
```

The user token's final representation is a weighted sum that **includes information from system tokens**. This is how the system prompt "influences" the user embeddings even after system tokens are cropped.

---

## Token Cropping: What, When, Why, How

### What Gets Cropped

After encoding, the **system prompt tokens are removed** from the output. Only user + assistant tokens remain.

### When It Happens

In `QwenImageTEModel.encode_token_weights()`:
```python
def encode_token_weights(self, token_weight_pairs, template_end=-1):
    # Full forward pass - system tokens attend during this
    out, pooled, extra = super().encode_token_weights(token_weight_pairs)

    # Find template_end: where system section ends
    tok_pairs = token_weight_pairs["qwen25_7b"][0]
    count_im_start = 0
    if template_end == -1:
        for i, v in enumerate(tok_pairs):
            elem = v[0]
            if isinstance(elem, numbers.Integral):
                if elem == 151644 and count_im_start < 2:  # <|im_start|>
                    template_end = i
                    count_im_start += 1

        # Adjust for "user\n" tokens after <|im_start|>
        if out.shape[1] > (template_end + 3):
            if tok_pairs[template_end + 1][0] == 872:  # "user"
                if tok_pairs[template_end + 2][0] == 198:  # "\n"
                    template_end += 3

    # CROP: Remove system tokens from output
    out = out[:, template_end:]
    extra["attention_mask"] = extra["attention_mask"][:, template_end:]
```

### Why Crop?

1. **DiT expects user content only:** The video model was trained on user descriptions, not system prompts
2. **Token efficiency:** Keeping system tokens would waste sequence length budget
3. **The influence is already baked in:** Through attention, system context is encoded into user embeddings

### How the Index is Calculated

For HunyuanVideo 1.5 with default template:
```
Position 0:   <|im_start|>     # token 151644 - first occurrence
Position 1:   system
Position 2:   \n
Position 3-N: [system prompt content]
Position N+1: <|im_end|>
Position N+2: \n
Position N+3: <|im_start|>     # token 151644 - second occurrence (user section)
Position N+4: user             # token 872
Position N+5: \n               # token 198
Position N+6: [user content starts here]

template_end = N + 6  (after "user\n")
```

The cropped output starts at `template_end`, meaning only user content and assistant marker remain.

---

## Qwen2.5-VL Architecture Details

### Grouped Query Attention (GQA)

```python
# Config
num_attention_heads = 28  # Query heads
num_key_value_heads = 4   # KV heads (shared across query groups)

# In attention:
xk = xk.repeat_interleave(28 // 4, dim=1)  # Expand 4 KV heads to 28
xv = xv.repeat_interleave(28 // 4, dim=1)
```

Each query head attends using shared KV heads, reducing memory by 7x.

### M-RoPE (Multimodal Rotary Position Embedding)

For text-only, standard 1D positions:
```python
position_ids = torch.arange(0, seq_len)
```

For vision+text, 3D positions (t, h, w):
```python
# rope_dims = [16, 24, 24] for Qwen2.5-VL
# Position IDs are 3xN tensor for multimodal
position_ids[0, :] = temporal_positions
position_ids[1, :] = height_positions
position_ids[2, :] = width_positions
```

### Layer Structure

```python
for layer in layers:
    # Pre-norm architecture
    residual = x
    x = input_layernorm(x)
    x = self_attention(x, attention_mask, freqs_cis)
    x = residual + x

    residual = x
    x = post_attention_layernorm(x)
    x = mlp(x)  # SwiGLU activation
    x = residual + x
```

### MLP (SwiGLU)

```python
def forward(self, x):
    # Gate and up projection
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    # SwiGLU: silu(gate) * up
    hidden = torch.nn.functional.silu(gate) * up
    # Down projection
    return self.down_proj(hidden)
```

---

## byT5 Architecture and Quote Extraction

### Quote Detection

```python
# In HunyuanImageTokenizer.tokenize_with_weights():
pattern_quote_double = r'\"(.*?)\"'
pattern_quote_chinese_single = r''(.*?)''
pattern_quote_chinese_double = r'"(.*?)"'

matches = re.findall(pattern_quote_double, text)
matches += re.findall(pattern_quote_chinese_single, text)
matches += re.findall(pattern_quote_chinese_double, text)

if len(matches) > 0:
    # Format for byT5: 'Text "Hello". Text "World". '
    formatted = ''.join(f'Text "{m}". ' for m in matches)
    tokens['byt5'] = self.byt5.tokenize_with_weights(formatted)
```

### byT5 Encoding

byT5 operates on **raw UTF-8 bytes**, not subword tokens:
```
"Hello" -> [72, 101, 108, 108, 111]  # ASCII bytes
"   " -> [228, 184, 150, 231, 149, 140]  # UTF-8 bytes for Chinese
```

### T5 Attention with Relative Position Bias

Unlike RoPE (absolute positions), T5 uses **learned relative position biases**:

```python
def compute_bias(self, query_length, key_length, device):
    # Relative position: memory_pos - query_pos
    relative_position = memory_position - context_position

    # Bucket into 32 bins (logarithmic for large distances)
    bucket = self._relative_position_bucket(relative_position)

    # Look up learned bias for each bucket
    bias = self.relative_attention_bias(bucket)  # (q_len, k_len, heads)
    return bias.permute([2, 0, 1]).unsqueeze(0)
```

This allows T5 to generalize to sequences longer than training.

### byT5 Output Integration

```python
# In HunyuanImageTEModel.encode_token_weights():
if self.byt5_small is not None and "byt5" in token_weight_pairs:
    byt5_out = self.byt5_small.encode_token_weights(token_weight_pairs["byt5"])
    extra["conditioning_byt5small"] = byt5_out[0]
```

The byT5 embeddings are passed separately to the DiT, which has dedicated cross-attention layers for text rendering.

---

## Vision Token Processing

### Image Token Detection

```python
# Token 151655 = <|image_pad|>
for i in range(len(tokens)):
    if tokens[i][0] == 151655:
        if len(images) > embed_count:
            tokens[i] = ({"type": "image", "data": images[embed_count]}, tokens[i][1:])
            embed_count += 1
```

### Vision Preprocessing

```python
# In Qwen25_7BVLI.preprocess_embed():
def preprocess_embed(self, embed, device):
    if embed["type"] == "image":
        image, grid = qwen_vl.process_qwen2vl_images(embed["data"])
        return self.visual(image.to(device), grid), grid
```

The vision transformer produces embeddings that replace `<|image_pad|>` tokens in the sequence.

### 3D Position Encoding for Vision

```python
# Images get 3D positions for M-RoPE
position_ids = torch.zeros((3, seq_len))
position_ids[0, image_start:image_end] = temporal_start  # Same time position
position_ids[1, image_start:image_end] = height_positions  # Row positions
position_ids[2, image_start:image_end] = width_positions   # Column positions
```

---

## Final Conditioning Construction

### Output Shape

```
Qwen output: [batch, seq_len - template_end, 3584]
byT5 output: [batch, quote_tokens, 1472] (optional)
```

### ComfyUI Conditioning Format

```python
conditioning = [
    [
        tensor,  # Shape: [1, N, 3584]
        {
            "attention_mask": mask,  # Optional, for padded sequences
            "conditioning_byt5small": byt5_tensor  # Optional, for text rendering
        }
    ]
]
```

### Positive vs Negative Encoding

Both use the **same system template**:
```python
# Positive
positive_cond = encode(system_prompt, user_prompt)

# Negative (same template for consistent embedding space)
negative_cond = encode(system_prompt, "low quality, blurry, distorted...")
```

Using the same system template ensures positive and negative embeddings are in the same semantic space for CFG.

---

## Practical Implications

### 1. System Prompts DO Affect Output

Even though system tokens are cropped, they influence user embeddings through attention. A structured system prompt like:
```
"Describe the video by detailing:
1. Main content and theme
2. Colors, shapes, textures
3. Actions and temporal changes
4. Background and atmosphere
5. Camera angles and movements"
```
...guides the model to produce embeddings that encode this structure.

### 2. Custom Templates Need Chat Markers

If you provide a custom system prompt, format it with `<|im_start|>` markers so ComfyUI doesn't double-wrap:
```python
formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
```

### 3. Quote Handling for Text Rendering

Put text you want rendered precisely in quotes:
```
"A sign saying \"OPEN 24 HOURS\" on a storefront"
```
This triggers byT5 encoding for accurate text generation.

### 4. Token Budget Awareness

After cropping, you have ~95 fewer tokens for your content (system template size). Long prompts may get truncated.

### 5. Negative Prompts Share System Context

Both positive and negative prompts see the same system template through attention, ensuring they're semantically aligned for CFG subtraction.

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT                                        │
│  system_prompt + user_prompt + (optional quotes)                    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      TEMPLATE FORMATTING                             │
│  <|im_start|>system\n{system}<|im_end|>\n                           │
│  <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                              ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│       QWEN2.5-VL            │    │         byT5                │
│  Tokenize full template     │    │  Tokenize quoted text only  │
│  28 transformer layers      │    │  T5 encoder layers          │
│  GQA + M-RoPE attention     │    │  Relative position bias     │
│  ─────────────────────────  │    │  ─────────────────────────  │
│  Output: [batch, N, 3584]   │    │  Output: [batch, M, 1472]   │
└─────────────────────────────┘    └─────────────────────────────┘
                   │                              │
                   ▼                              │
┌─────────────────────────────┐                   │
│       TOKEN CROPPING        │                   │
│  Remove system tokens       │                   │
│  Keep user + assistant      │                   │
│  System influence retained  │                   │
│  via attention mechanism    │                   │
└─────────────────────────────┘                   │
                   │                              │
                   └──────────────┬───────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       CONDITIONING                                   │
│  Primary: Qwen embeddings (semantic content)                        │
│  Secondary: byT5 embeddings (text rendering, optional)              │
│  Attention mask (for padded sequences)                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                          [DiT / Video Model]
```

The key insight: **System prompts work through attention influence, not presence.** Even after cropping, the user embeddings "remember" the system context because they attended to it during encoding.
