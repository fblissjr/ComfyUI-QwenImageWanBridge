# Z-Image Character Generation Guide

Using structured character profiles with multi-turn conversations to maintain consistency while making precise edits.

**New to Z-Image nodes?** Start with [z_image_intro.md](z_image_intro.md) for the basics.

---

## Multi-Turn Character Editing

The most powerful use of our nodes is maintaining character consistency across edits. By defining a detailed character in the first turn, then making targeted modifications in subsequent turns, you can make precise changes while preserving the rest.

### Full Example: Walter Finch

**ZImageTextEncoder (First Turn):**
```
system_prompt: "Generate an image in classic American comic book style. Bold outlines, flat colors with halftone shading."

user_prompt: |
  # Character Profile: Walter Finch (Wally)

  ## Core Identity
  - **Name:** Walter Finch (Nickname: Wally)
  - **Gender:** Male
  - **Age:** 72
  - **Ethnicity:** Caucasian (British descent)
  - **Skin Tone:** Fair, warm sun-kissed glow on cheeks and nose

  ## Head & Face
  - **Face Shape:** Oval, prominent jawline, receding hairline
  - **Eye Color:** Ice-blue with subtle gold flecks around the iris
  - **Hair:** Pure white, side-parted, full beard and mustache
  - **Glasses:** Thin, gold-rimmed reading glasses
  - **Expression:** Warm, gentle smile with a twinkle in his eye

  ## Attire
  - Light gray and blue checkered button-down shirt
  - Dark grey wool trousers
  - Brown leather loafers

  ## Props
  - Pale lavender ceramic mug of coffee
  - Gold pocket watch chain visible in breast pocket

add_think_block: true
thinking_content: "Hmm lets make his beard a little red and keep everything else the same."
assistant_content: "Sure, here's a photo of Wally with his red and white beard."
```

**ZImageTurnBuilder (Second Turn):**
```
previous: [from encoder's conversation output]
clip: [from CLIPLoader - enables direct conditioning output]

user_prompt: "Let's put a cute baby flying sloth above him too"
thinking_content: "Hm, let's make it a black sloth, flying above the man's head."
is_final: true
```

### Why This Works

The full conversation is encoded together:
```
<|im_start|>system
Generate an image in classic American comic book style...<|im_end|>
<|im_start|>user
# Character Profile: Walter Finch (Wally)
[full character sheet]<|im_end|>
<|im_start|>assistant
<think>Hmm lets make his beard a little red...</think>
Sure, here's a photo of Wally with his red and white beard.<|im_end|>
<|im_start|>user
Let's put a cute baby flying sloth above him too<|im_end|>
<|im_start|>assistant
<think>Hm, let's make it a black sloth, flying above the man's head.</think>

```

The model sees the original detailed definition AND all the subsequent modifications in context.

### Tips for Multi-Turn Character Work

1. **Be explicit about what stays**: In thinking blocks, note what should be preserved
2. **Make one change per turn**: Don't overload modifications
3. **Use thinking to guide changes**: "Keep the glasses and shirt, only change the beard"
4. **Chain carefully**: Each turn builds on previous ones

---

## Using LLMs to Generate Prompts

You can use any LLM to help generate detailed character profiles. However, there's a significant technical advantage to using **Qwen3 family models** (Qwen3-0.5B through Qwen3-235B) when generating prompts for Z-Image.

### Why Qwen3 Models Are Optimal

Z-Image uses Qwen3-4B as its text encoder. All models in the Qwen3 family share the **same tokenizer and vocabulary**. This creates a unique advantage.

#### The Problem: Cross-Family "Translation"

When you use a different model family (Llama, GPT, Claude, etc.) to generate a prompt:

1. **Source Generation**: The LLM generates text using its own tokenizer
2. **Re-tokenization**: Z-Image's encoder (Qwen3-4B) re-encodes that text into its own token IDs
3. **Information Loss**: The subtle semantic nuances intended by the source model are lost in translation

Different tokenizers break text into different subword units. The same sentence becomes different sequences of integers. The receiving model gets a "translated" version of the concept, not the original intent.

#### The Solution: Direct Token Transfer

When you use a Qwen3 model to generate the prompt:

1. **Source Generation**: Qwen3 (any size) generates text as a sequence of **Qwen3 Token IDs**
2. **Direct Transfer**: These exact Token IDs pass directly to Z-Image's encoder (Qwen3-4B)
3. **High-Fidelity Priming**: The encoder interprets the incoming tokens with near-perfect fidelity

Because both models share the same vocabulary, the exact integer ID representing each subword is identical. You're not transferring text - you're transferring a pre-computed semantic state.

#### The Thinking Block Advantage

This is especially powerful with `<think>` blocks. When a larger Qwen3 model (like Qwen3-72B or Qwen3-235B) generates thinking content:

```
<think>
The character should have weathered skin with visible sun damage around the eyes.
The lighting should be warm side-light from the left, suggesting late afternoon.
Expression is contemplative but not sad - slight smile, distant gaze.
</think>
```

These tokens prime Z-Image's encoder to a configuration that directly reflects the larger model's reasoning. The 4B encoder essentially receives "art direction" from a much more capable model, in a format it natively understands.

### Practical Recommendations

| Scenario | Recommendation |
|----------|----------------|
| Quick prompts | Any LLM is fine |
| Detailed character work | Prefer Qwen3 family |
| Complex thinking blocks | Strongly prefer Qwen3-72B+ |
| Maximum fidelity | Qwen3-235B-A22B (if available) |

### Example: Using Qwen3 for Prompt Generation

**System prompt for the generating LLM:**

```
You are a visual prompt engineer for a text-to-image model.

Your task is to generate detailed, visually-specific character descriptions.
Focus on concrete visual details - colors, textures, lighting, poses.
Avoid abstract concepts like "beautiful" or "mysterious".

When asked about a character:
1. First, think through the visual elements in a <think> block
2. Then provide a structured character profile

Format your thinking to include:
- Key distinguishing features
- Lighting and atmosphere decisions
- Composition notes
- What to emphasize vs de-emphasize

The output will be used directly as a prompt, so be specific and visual.
```

**User request:**
```
Create a character profile for a 72-year-old British gentleman named Walter,
in American comic book style.
```

**Qwen3 output (use this directly in ZImageTextEncoder):**

```
<think>
Key features: elderly but dignified, British character actor type.
Face: oval with prominent jawline, weathered but kind.
Eyes should be distinctive - ice blue with warmth, gold-rimmed glasses.
Hair: pure white, well-groomed beard, side-parted hair.
Attire: classic intellectual - checkered shirt, wool trousers, leather loafers.
Props: coffee mug (lavender ceramic), pocket watch chain - these add character.
Style: bold comic outlines, flat colors, halftone shading.
Expression: warm smile, twinkle in eye - approachable grandfather type.
Lighting: even comic-style, minimal shadows, clear features.
</think>

# Character Profile: Walter Finch (Wally)
[... structured profile follows ...]
```

The entire output - including the `<think>` block - can be pasted directly into `user_prompt` with `add_think_block: true`. The thinking content goes into `thinking_content`, and the model receives the full semantic context.

### Using Other LLMs

Other LLMs (Claude, GPT, Llama, etc.) can absolutely generate good prompts. The structured formats in this guide work with any source. You'll just miss the token-level fidelity advantage.

If using a non-Qwen LLM:
- Focus on the visual vocabulary sections below
- Use clear, concrete language
- The prompt quality matters more than the source model for most use cases

The Qwen3 advantage is most noticeable for complex, nuanced character work where subtle semantic details matter.

---

## Core Principle

The model needs **visual details**, not story. Every word should describe something you can see.

| Good (Visual) | Bad (Abstract) |
|---------------|----------------|
| deep-set eyes with dark circles | tired expression |
| worn brown leather jacket, cracked at elbows | old jacket |
| sharp cheekbones, hollow cheeks | thin face |
| warm amber side-light from left | good lighting |

---

## Format Options

### Option 1: Prose (Most Natural)

Natural sentences flow well. The model reads this like a description.

```
A woman in her late 20s with sharp East Asian features. Black hair cut asymmetrically -
short on the right, falling past her jaw on the left. Pale skin with a slight blue
undertone, dark circles under her eyes. She wears a oversized grey hoodie, hood down,
with a faded band logo barely visible. Expression is guarded but curious, mouth slightly
open as if about to speak. Soft diffused light from above, cool color temperature.
Background is out of focus warm tones suggesting an interior space.
```

### Option 2: Sectioned Prose (Organized but Natural)

Headers organize without fragmenting.

```
FACE: Sharp East Asian features, late 20s. Pale skin with blue undertone, visible
dark circles. High cheekbones, small nose, full lips slightly parted.

HAIR: Black, asymmetrical cut. Cropped close on right side, falls to jaw on left.
Straight, slightly greasy texture.

EXPRESSION: Guarded curiosity. Eyes alert and watchful, brows slightly raised.
Mouth open as if about to speak.

CLOTHING: Oversized grey hoodie, hood down. Fabric is soft cotton, well-worn.
Faded band logo on chest, text illegible.

LIGHTING: Soft diffused overhead light, cool temperature. Subtle warm fill
from below suggesting screen glow.
```

### Option 3: Compact List (Quick Reference)

Dense but specific. Good for simpler characters.

```
- late 20s woman, sharp East Asian features
- asymmetric black hair: short right, jaw-length left
- pale skin, blue undertone, dark circles
- oversized grey hoodie, faded logo, hood down
- guarded expression, alert eyes, parted lips
- cool overhead light, warm fill from below
```

---

## Visual Vocabulary

### Skin
- **Tone**: warm ivory, cool beige, deep brown, olive, ruddy, sallow, ashen
- **Texture**: smooth, weathered, freckled, scarred, pocked, matte, dewy
- **Details**: crow's feet, laugh lines, acne scars, sun damage, stubble shadow

### Eyes
- **Shape**: almond, round, hooded, deep-set, wide-set, close-set, upturned
- **Color**: steel grey, warm brown, pale blue, amber, heterochromia
- **State**: bloodshot, glassy, sharp, tired, bright, narrowed, wide

### Hair
- **Texture**: straight, wavy, curly, coily, frizzy, slicked, matte, glossy
- **State**: clean, greasy, windswept, disheveled, meticulous, damp
- **Details**: roots showing, split ends, grey streaks, highlights, undercut

### Clothing Condition
- **New**: crisp, pressed, tags-on, bright colors
- **Worn**: faded, soft, stretched, pilled, comfortable
- **Damaged**: torn, patched, stained, frayed, threadbare

### Lighting
- **Direction**: overhead, side (left/right), below, behind (rim), frontal
- **Quality**: harsh/hard, soft/diffused, dappled, even, dramatic
- **Color**: warm (golden, amber), cool (blue, white), neutral, mixed

---

## Templates by Shot Type

### Portrait (Head/Shoulders)

```
SUBJECT: [age] [gender] with [ethnicity/features]. [face shape], [skin description].

EYES: [color], [shape]. [current state/expression].

HAIR: [color], [texture], [style]. [current state].

DISTINGUISHING: [1-2 notable features: scar, piercing, glasses, makeup].

EXPRESSION: [emotion]. [specific facial details that show it].

LIGHTING: [direction] [quality] light, [color temperature]. [background].
```

**Example:**
```
SUBJECT: Mid-30s man with Mediterranean features. Angular face, olive skin with
visible stubble shadow and sun-weathered texture around eyes.

EYES: Dark brown, deep-set. Focused, slightly narrowed, crow's feet visible.

HAIR: Black with grey at temples, short and neat but finger-combed. Clean fade.

DISTINGUISHING: Thin white scar through left eyebrow. Small gold hoop in left ear.

EXPRESSION: Calm assessment. Slight asymmetric smile, more smirk than grin.
Relaxed jaw, direct gaze.

LIGHTING: Warm side light from camera left, soft fill from right. Cool grey
background, shallow depth of field.
```

### Full Body

```
BUILD: [height impression] [body type]. [posture].

FACE: [2-3 key features]. [expression].

HAIR: [brief - color, length, state].

UPPER: [base layer]. [outer layer if any]. [condition/details].

LOWER: [pants/skirt type]. [condition]. [fit].

FEET: [footwear]. [condition].

ACCESSORIES: [visible items].

POSE: [stance]. [arm position]. [weight]. [direction facing].

SETTING: [environment]. [lighting]. [atmosphere].
```

**Example:**
```
BUILD: Tall and lean, almost gangly. Shoulders slightly hunched, hands in pockets.

FACE: Long face, prominent nose, wide mouth. Tired half-smile.

HAIR: Sandy brown, shaggy, needs a cut. Falls into eyes.

UPPER: Faded black t-shirt under unbuttoned flannel, red and grey plaid.
Sleeves rolled to elbows. Both well-worn, soft fabric.

LOWER: Dark jeans, slim fit, worn white at knees. Sitting low on hips.

FEET: White canvas sneakers, dirty, laces loose.

ACCESSORIES: Leather watch with scratched face. Headphones around neck.

POSE: Standing with weight on left leg, right knee bent. Left hand in
front pocket, right holding phone at side. Facing camera, head tilted.

SETTING: Urban street corner, afternoon. Overcast daylight, flat and even.
Blurred storefronts behind, warm tones.
```

### Action Shot

```
ACTION: [what's happening]. [intensity]. [direction of movement].

BODY: [position in motion]. [which limbs lead]. [tension points].

FACE: [expression matching effort]. [hair movement].

CLOTHING: [how fabric responds to motion]. [what's visible].

MOTION CUES: [blur elements]. [environmental reaction]. [dynamic elements].

CAMERA: [angle]. [distance]. [what's in focus].
```

**Example:**
```
ACTION: Mid-sprint, pushing off right foot. Explosive, desperate. Moving
left to right across frame.

BODY: Right leg extended behind, left knee driving up. Right arm back,
left arm forward. Torso leaning into run, shoulders rotated.

FACE: Teeth gritted, eyes focused ahead. Black hair streaming behind,
off her face.

CLOTHING: Red athletic jacket unzipped, flying open. Black tank underneath.
Grey leggings. White sneakers, right toe pushing off ground.

MOTION CUES: Hair and jacket trailing. Slight motion blur on trailing arm.
Dust kicked up behind right foot.

CAMERA: Low angle, knee height. Medium shot, full body visible.
Focus on face and lead leg, background blurred.
```

---

## Density Levels

**Minimal** (for simple/stylized):
```
Young woman, short pink hair, oversized denim jacket, confident smirk,
warm golden hour light
```

**Standard** (balanced detail):
```
SUBJECT: Young woman, early 20s, round face with soft features
HAIR: Bubblegum pink, choppy pixie cut, slightly messy
CLOTHING: Oversized vintage denim jacket, pins on lapel, white tee underneath
EXPRESSION: Confident smirk, one eyebrow raised, direct eye contact
LIGHTING: Golden hour side light from right, warm tones, soft shadows
```

**Detailed** (maximum control):
```
SUBJECT: Young woman, early 20s. Round face with soft features, button nose,
full cheeks with subtle blush. Fair skin with light freckles across nose bridge.
Small beauty mark below left eye.

EYES: Bright green, round and large. Playful expression, slight squint from
smiling. Dark mascara, subtle winged liner.

HAIR: Bubblegum pink, intentionally faded at roots showing natural brown.
Choppy pixie cut, longer on top, textured and messy. Pieces falling across forehead.

EXPRESSION: Confident smirk pulling right corner of mouth up. One eyebrow
raised. Direct challenging eye contact. Relaxed jaw, neck slightly tilted.

CLOTHING: Oversized vintage denim jacket, light wash, worn soft. Sleeves
pushed up to elbows. Collection of enamel pins on left lapel. Plain white
crew neck tee visible underneath, slightly oversized.

LIGHTING: Golden hour sunlight from camera right, warm orange tones.
Soft shadows on left side of face. Hair catching light, almost glowing.
Background blown out, creamy warm bokeh suggesting outdoor setting.
```

---

## Quick Reference Card

**Always include:**
1. Age/gender/ethnicity cues
2. 2-3 specific facial features
3. Hair color + style + state
4. Lighting direction + quality

**For portraits add:**
- Eye color and expression
- Distinguishing marks
- Background treatment

**For full body add:**
- Body type and posture
- Complete outfit with condition
- Pose specifics
- Environment

**For action add:**
- Movement direction
- Which body parts lead
- Fabric/hair motion
- Camera angle

---

## Anti-Patterns

| Avoid | Why | Instead |
|-------|-----|---------|
| "beautiful" | Subjective, no visual info | Describe specific features |
| "mysterious expression" | Abstract | "narrowed eyes, slight frown, closed lips" |
| "nice clothes" | Vague | "navy wool peacoat, brass buttons" |
| "good lighting" | Meaningless | "soft side light from left, warm tone" |
| Long backstory | Model can't visualize story | Cut to visual details only |
| "like [celebrity]" | Inconsistent results | Describe actual features |

---

## Workflow Summary

1. **Choose density** - minimal for stylized, detailed for photorealistic
2. **Pick format** - prose flows naturally, sections organize complex characters
3. **Use visual vocabulary** - concrete colors, textures, states
4. **Include lighting** - it defines the entire mood
5. **Match to template** - portrait spec + portrait template
