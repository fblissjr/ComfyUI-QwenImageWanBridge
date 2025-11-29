# Z-Image Nodes: What Is This?

## The 30-Second Answer

**Q: What does this do?**
It replaces ComfyUI's stock text encoder for Z-Image and gives you control over how your prompt is structured before it hits the model.

**Q: Why would I need this?**
Z-Image uses Qwen3-4B as its text encoder - an LLM that understands conversations, system instructions, and "thinking" processes. Our nodes let you use those capabilities instead of ignoring them.

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

We give you the full template:
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

Now you control:
- **System prompt**: Set the artistic direction, style, constraints
- **User prompt**: Your actual request
- **Thinking content**: Guide the model's "reasoning" about how to approach the image
- **Assistant content**: Prime the model's response direction

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
| `add_think_block` | Enable `<think>` structure |
| `thinking_content` | Content inside think tags |
| `assistant_content` | Model's response after thinking |
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
| `user_prompt` | The next user message |
| `thinking_content` | Assistant's thinking for this turn |
| `assistant_content` | Assistant's response |
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

### Level 3: With Thinking

Guide the model's interpretation.

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

Prime the model's output direction.

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

### Level 5: Multi-Turn Conversation

Chain turns to simulate iterative refinement.

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

### Why This Works

The conversation format lets you:
1. Establish a detailed "ground truth" in the first message
2. Make surgical edits in subsequent messages
3. Use the thinking block to guide how changes should be applied
4. Keep the model focused on modifications rather than regenerating from scratch

This is experimental - we're essentially using the LLM's conversation understanding to guide image generation consistency. Results vary, but it's a powerful technique for character work.

---

## Templates

We include 100+ templates in `nodes/templates/z_image/`. Some highlights:

| Template | Description |
|----------|-------------|
| `photorealistic` | Natural lighting and realistic details |
| `comic_american` | Bold outlines, flat colors, dynamic poses |
| `anime_ghibli` | Studio Ghibli watercolor style |
| `neon_cyberpunk` | Neon lights, rain-slicked streets |
| `oil_painting_classical` | Renaissance master technique |
| `pixel_art` | Retro 8/16-bit aesthetic |
| `character_design` | Turnaround sheets, model references |

Templates auto-fill the `system_prompt` field. You can edit it after selection.

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
Experimentally, yes. It seems to guide composition and interpretation. Think of it as "art direction notes."

**Q: Can I use this for other models?**
The ZImageTextEncoder is specifically for Z-Image (Qwen3-4B encoder). The PromptKeyFilter utility works with any text encoder.

**Q: What's the `strip_key_quotes` for?**
When using JSON-formatted prompts (from LLMs), the quoted keys like `"subject":` can appear as literal text in images. This filter strips the quotes from keys while preserving values.

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
| Guide interpretation | Add thinking_content |
| Iterative edits | ZImageTurnBuilder chain |
| Character consistency | Structured prompt + multi-turn |
| Full control | raw_prompt mode |

The nodes are just building an LLM message chain. The magic is in how you structure that chain.
