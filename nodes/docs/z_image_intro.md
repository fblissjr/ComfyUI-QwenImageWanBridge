# Z-Image Nodes: What Is This?

## The 30-Second Answer

**Q: What does this do?**
It replaces ComfyUI's stock text encoder for Z-Image and gives you control over how your prompt is structured before it hits the model.

**Q: Why would I need this?**
Z-Image was trained on prompts formatted as Qwen3-4B chat conversations (with system/user/assistant roles and special tokens). Our nodes let you format your prompts to match that training format, instead of sending plain text.

**Q: Is this a new model?**
No. Same Z-Image model, same weights. We're just giving you access to the full prompt format the model was trained on.

**Q: Should I bother?**
- **Just want to generate images?** The stock encoder works fine. Skip this.
- **Want system prompts, templates, or more control?** Yes, use `ZImageTextEncoder`.
- **Want to simulate iterative edits or character consistency?** Yes, use the Turn Builder for multi-turn conversations.

---

## Quick Start

### Simplest Usage (matches stock behavior)
```
CLIPLoader (lumina2) -> ZImageTextEncoder -> KSampler
                        user_prompt: "A cat sleeping"
```

### With a System Prompt
```
CLIPLoader -> ZImageTextEncoder -> KSampler
              template_preset: photorealistic
              user_prompt: "A cat sleeping"
```

That's it. Everything else is optional.

---

## What's Actually Happening

### The Problem with Stock Encoding

ComfyUI's default encoder treats your prompt as raw text:
```
A cat sleeping on a windowsill
```

But Z-Image was trained with Qwen3-4B's **chat template** - a structured conversation format. The model expects:
```
<|im_start|>user
A cat sleeping on a windowsill<|im_end|>
<|im_start|>assistant

```

When you use the stock encoder, ComfyUI adds this wrapper automatically. But that's ALL it adds. You get no system prompt, no thinking, no conversation history.

### What Our Nodes Add

We let you build the full template structure:
```
<|im_start|>system
Generate a photorealistic image with natural lighting.<|im_end|>
<|im_start|>user
A cat sleeping on a windowsill<|im_end|>
<|im_start|>assistant
<think>
Warm afternoon light, shallow depth of field, peaceful mood.
</think>

Here's the cozy scene you requested.
```

**Important:** There's no LLM running here. Our nodes are text formatters - they assemble your input into the chat template format, wrap it with special tokens, and pass it to the text encoder. The "thinking" and "assistant" content is whatever text YOU provide.

You control:
- **System prompt**: Text that goes in the system role position
- **User prompt**: Text that goes in the user role position
- **Thinking content**: Text placed inside `<think>` tags (you write this, not an LLM)
- **Assistant content**: Text placed after the think block

---

## Understanding Special Tokens

### What Are They?

Special tokens are markers that tell the model "this is structure, not content." They're defined in the model's `tokenizer_config.json`:

| Token | Purpose |
|-------|---------|
| `<\|im_start\|>` | Start of a message |
| `<\|im_end\|>` | End of a message |
| `system` | Role: instructions for the model |
| `user` | Role: the human's request |
| `assistant` | Role: the model's response |
| `<think>` | Start of reasoning (Qwen3 thinking mode) |
| `</think>` | End of reasoning |

### Why Do They Exist?

LLMs are trained on conversations. The special tokens let the model understand:
- Who's speaking (system vs user vs assistant)
- When a message ends
- What's instruction vs content

You can see the exact template in [Qwen3-4B's tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json).

### The Chat Template

From the tokenizer config:
```jinja
{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
<|im_start|>assistant
```

This is why Z-Image responds better to properly structured prompts - it was trained to expect this format.

---

## Node Reference

### ZImageTextEncoder

The main node. Handles a complete first turn of conversation.

**Key Inputs:**
| Input | What It Does |
|-------|--------------|
| `user_prompt` | Your generation request (required) |
| `template_preset` | Quick style selection (100+ templates) |
| `system_prompt` | Custom instructions (auto-filled by template) |
| `add_think_block` | Add `<think>` tags to the template |
| `thinking_content` | Your text to place inside think tags |
| `assistant_content` | Your text to place after think block |
| `raw_prompt` | Bypass everything, write your own tokens |
| `strip_key_quotes` | Clean JSON keys from LLM output |

**Outputs:**
| Output | What It's For |
|--------|---------------|
| `conditioning` | Connect to KSampler |
| `formatted_prompt` | See exactly what was encoded |
| `conversation` | Chain to TurnBuilder for multi-turn |

### ZImageTurnBuilder

Add conversation turns after the initial encoder. This is where it gets interesting.

**Key Inputs:**
| Input | What It Does |
|-------|--------------|
| `previous` | Connect from encoder or another TurnBuilder |
| `clip` | Optional - connect to output conditioning directly |
| `user_prompt` | Text for the next user message |
| `thinking_content` | Your text for the think block |
| `assistant_content` | Your text for the assistant response |
| `is_final` | Leave last message open for generation |

**Two Workflow Options:**

1. **Direct encoding** (connect clip to TurnBuilder):
```
ZImageTextEncoder -> TurnBuilder (clip connected) -> conditioning -> KSampler
```

2. **Chain back** (no clip on TurnBuilder):
```
ZImageTextEncoder -> TurnBuilder -> ZImageTextEncoder (conversation_override)
```

### ZImageTextEncoderSimple

Simplified encoder for quick use - ideal for **negative prompts**.

Same template/thinking support as the full encoder, but no conversation chaining. Just conditioning + formatted_prompt outputs.

**Usage:**
```
Positive: ZImageTextEncoder -> KSampler (positive)
Negative: ZImageTextEncoderSimple -> KSampler (negative)
         user_prompt: "bad anatomy, blurry, watermark, text"
```

Both use the same Qwen3-4B chat template format for consistency.

---

## Progressive Examples

### Level 1: Basic (No Template)

Just a user prompt. Matches diffusers default.

```
user_prompt: "A red apple on a wooden table"
```

**Encoded as:**
```
<|im_start|>user
A red apple on a wooden table<|im_end|>
<|im_start|>assistant

```

### Level 2: With System Prompt

Add artistic direction via template.

```
template_preset: photorealistic
user_prompt: "A red apple on a wooden table"
```

**Encoded as:**
```
<|im_start|>system
Generate a photorealistic image with accurate lighting, natural textures, and realistic details.<|im_end|>
<|im_start|>user
A red apple on a wooden table<|im_end|>
<|im_start|>assistant

```

### Level 3: With Think Block

Add text in the think block position (you write this text, not an LLM).

```
template_preset: photorealistic
user_prompt: "A red apple on a wooden table"
add_think_block: true
thinking_content: "Morning light from the left, shallow depth of field, rustic oak texture."
```

**Encoded as:**
```
<|im_start|>system
Generate a photorealistic image with accurate lighting, natural textures, and realistic details.<|im_end|>
<|im_start|>user
A red apple on a wooden table<|im_end|>
<|im_start|>assistant
<think>
Morning light from the left, shallow depth of field, rustic oak texture.
</think>

```

### Level 4: With Assistant Response

Add text in the assistant response position.

```
user_prompt: "A red apple on a wooden table"
thinking_content: "Morning light, shallow DOF, rustic oak."
assistant_content: "Here's a warm, inviting still life."
```

**Encoded as:**
```
<|im_start|>user
A red apple on a wooden table<|im_end|>
<|im_start|>assistant
<think>
Morning light, shallow DOF, rustic oak.
</think>

Here's a warm, inviting still life.<|im_end|>
```

Note: When you provide `assistant_content`, the message closes with `<|im_end|>`. When empty, it stays open (model continues generating).

### Level 5: Multi-Turn Format

Chain turns to build a longer conversation structure.

```
[ZImageTextEncoder]
user_prompt: "A portrait of an old man"
thinking_content: "Wise eyes, weathered skin, warm lighting."
assistant_content: "Here's a distinguished gentleman."

    |
    v

[ZImageTurnBuilder]
user_prompt: "Make his beard red"
thinking_content: "Keep everything else, just change beard color."
assistant_content: "Updated with a red beard."
clip: (connected for direct encoding)
```

**Full encoded conversation:**
```
<|im_start|>user
A portrait of an old man<|im_end|>
<|im_start|>assistant
<think>
Wise eyes, weathered skin, warm lighting.
</think>

Here's a distinguished gentleman.<|im_end|>
<|im_start|>user
Make his beard red<|im_end|>
<|im_start|>assistant
<think>
Keep everything else, just change beard color.
</think>

Updated with a red beard.
```

---

## Advanced: Character Consistency with Structured Prompts

This is where multi-turn really shines. You can define a detailed character sheet, then make precise edits while maintaining consistency.

See: [Character Generation Guide](z_image_character_generation.md) - includes visual vocabulary, templates by shot type, and using LLMs (especially Qwen3) to generate prompts.

### The Concept

1. **First turn**: Define your character with detailed structured data
2. **Subsequent turns**: Make targeted modifications

### Example: Walter Finch (Wally)

**ZImageTextEncoder (first turn):**
```
system_prompt: "Generate an image in classic American comic book style."

user_prompt: |
  # Character Profile: Walter Finch (Wally)

  ## Core Identity
  - **Name:** Walter Finch (Nickname: Wally)
  - **Age:** 72
  - **Ethnicity:** Caucasian (British descent)

  ## Head & Face
  - **Eye Color:** Ice-blue with subtle gold flecks
  - **Hair:** Pure white, side-parted, full beard
  - **Expression:** Warm, gentle smile

  ## Attire
  - Gray/blue checkered button-down shirt
  - Dark grey wool trousers
  - Brown leather loafers

  ## Props
  - Pale lavender ceramic mug of coffee
  - Gold pocket watch chain

thinking_content: "Hmm lets make his beard a little red and keep everything else the same."

assistant_content: "Sure, here's a photo of Wally with his red and white beard."
```

**ZImageTurnBuilder (second turn):**
```
user_prompt: "Let's put a cute baby flying sloth above him too"

thinking_content: "Hm, let's make it a black sloth, flying above the man's head."
```

**Result:** The model maintains Wally's detailed appearance while adding the requested modifications, because the full conversation context is preserved in the encoding.

### Why This Might Work

Z-Image was trained on conversation-formatted prompts. The theory is that structuring your prompt as a multi-turn conversation - where earlier turns establish details and later turns request modifications - might help the model understand what to preserve vs. change.

**To be clear:** We're not running an LLM here. We're formatting text to look like a conversation, then encoding it. Whether this actually helps consistency is experimental. The model may or may not interpret the conversation structure the way a chat LLM would. Results vary.

---

## Templates

We include 140+ templates in `nodes/templates/z_image/`. Some highlights:

**Style Templates:**
| Template | Description |
|----------|-------------|
| `photorealistic` | Natural lighting and realistic details |
| `comic_american` | Bold outlines, flat colors, dynamic poses |
| `anime_ghibli` | Studio Ghibli watercolor style |
| `neon_cyberpunk` | Neon lights, rain-slicked streets |
| `oil_painting_classical` | Renaissance master technique |
| `pixel_art` | Retro 8/16-bit aesthetic |
| `character_design` | Turnaround sheets, model references |

**Structured Prompt Templates (v2.9.10+):**
| Template | Description |
|----------|-------------|
| `json_structured` | Parse JSON-formatted prompts from LLMs |
| `yaml_structured` | Parse YAML hierarchical prompts |
| `markdown_structured` | Parse Markdown-formatted prompts |

These structured templates include pre-written text for the thinking block that instructs the model to extract visual concepts from structured data.

### Extended Template Format

Templates can now pre-fill multiple fields (not just system prompt):
- `system_prompt` - always filled from template body
- `add_think_block` - auto-enable the thinking checkbox
- `thinking_content` - pre-fill thinking content
- `assistant_content` - pre-fill assistant content

When you select a template, all configured fields get filled. You can edit any field before generation.

---

## Raw Mode

For complete control, use `raw_prompt` to bypass all formatting:

```
raw_prompt: |
  <|im_start|>system
  You are a surrealist painter.<|im_end|>
  <|im_start|>user
  A melting clock<|im_end|>
  <|im_start|>assistant
  <think>
  Dali-esque, desert landscape, impossible physics.
  </think>

  A dreamscape emerges.
```

When `raw_prompt` is set, all other fields are ignored. You're responsible for correct token formatting.

---

## Debugging

**See what's being encoded:**
- Connect `formatted_prompt` output to a Preview node
- Check server console for `[Z-Image] Formatted prompt:` logs

**Debug output includes:**
- Mode (direct/raw/conversation_override)
- Character counts for each field
- Token estimate
- Think tag verification

---

## FAQ

**Q: Does the system prompt actually affect the image?**
Yes, but subtly. It's most effective for style direction and constraints. The user prompt has the strongest influence.

**Q: Does the thinking content matter?**
Maybe. The text you put in the think block becomes part of the encoded prompt. Whether it influences the output depends on what Z-Image learned during training. Think of it as extra text in a specific position - not actual LLM reasoning.

**Q: Can I use this for other models?**
The ZImageTextEncoder is specifically for Z-Image (Qwen3-4B encoder). The PromptKeyFilter utility works with any text encoder.

**Q: What's the `strip_key_quotes` for?**
When using JSON-formatted prompts (from LLMs), the quoted text like `"subject": "a cat"` can appear as literal text in images. This filter strips all double quotes to prevent that.

**Q: Can I use an LLM to generate prompts?**
Yes. Any LLM works, but **Qwen3 family models** have a technical advantage: they share the same tokenizer as Z-Image's encoder (Qwen3-4B). This means tokens transfer directly without re-encoding, preserving subtle semantic nuances. See [Character Generation Guide](z_image_character_generation.md#using-llms-to-generate-prompts) for details.

---

## Files

- **Encoder code:** `nodes/z_image_encoder.py`
- **Templates:** `nodes/templates/z_image/`
- **Full documentation:** [z_image_encoder.md](z_image_encoder.md)
- **Character guide:** [z_image_character_generation.md](z_image_character_generation.md)
- **Example workflows:** `example_workflows/z-image_*.json`

---

## Summary

| You Want | Use This |
|----------|----------|
| Quick generation | Stock encoder or basic ZImageTextEncoder |
| Style control | ZImageTextEncoder + template_preset |
| Extra prompt text | Add thinking_content |
| Iterative edits | ZImageTurnBuilder chain |
| Character consistency | Structured prompt + multi-turn |
| **Negative prompts** | ZImageTextEncoderSimple |
| Full control | raw_prompt mode |

These nodes are text formatters - they build a conversation-shaped string and encode it. No LLM reasoning happens here.
