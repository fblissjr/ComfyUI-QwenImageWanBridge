# HunyuanVideo 1.5 Prompting Experiments Guide

A comprehensive guide to experimenting with system prompts, user prompts, and negative prompts for HunyuanVideo 1.5 text-to-video generation.

## Highlights

### Part 1: System Prompt Experiments
- Template showdown (compare cinematic vs documentary vs abstract)
- System prompt length testing (minimal vs verbose)
- Role-based personas (Director, Physicist, Animator, Poet)

### Part 2: User Prompt Experiments
- Component ablation (what happens when you remove shot type, camera, timing?)
- Motion vocabulary tests (walks vs strolls vs trudges vs glides)
- Camera movement vocabulary (pan, tilt, dolly, orbit, crane, handheld)
- Timing precision tests (vague vs "over 4 seconds" vs "97 frames")
- Multi-subject coordination

### Part 3: Negative Prompt Experiments
- Specificity tests (minimal vs comprehensive)
- Contradictory negatives (what if you negate the subject?)
- Style negatives (push toward/away from specific aesthetics)

### Part 4: CFG Scale Experiments
- CFG sweep (1.0 to 15.0)
- CFG + negative strength combinations

### Part 5: byT5 Multilingual Text
- Text rendering accuracy tests
- Multiple quoted strings
- Mixed languages (Chinese, Japanese, Korean, Arabic)
- Quote edge cases

### Part 6: Edge Cases
- Physics-breaking prompts
- Contradictory instructions
- Extreme prompt lengths
- Impossible cinematography

### Part 7-8: Testing Protocols and Fun Experiments
- The Genre Gauntlet (same action in 5 genres)
- The Speed Run (slow-mo vs normal vs timelapse)
- The Camera Challenge (10 different camera moves on one subject)
- The Impossible Shot (bullet-time, continuous underwater-to-sky)

---

## How Text Encoding Actually Works

Before experimenting, understand the pipeline:

```
System Prompt + User Prompt → Tokenize → Attention (together!) → Crop System → User Embeddings
```

**Key insight**: System tokens are encoded WITH user tokens (they attend to each other), then cropped. The "influence" persists in the user embeddings even though system tokens are removed.

This means your system prompt actually matters - it's not just metadata.

---

## Part 1: System Prompt Experiments

### The Five Pillars (Default Template)

The default HunyuanVideo template instructs the model to describe:
1. Main content and theme
2. Visual attributes (color, shape, size, texture)
3. Temporal dynamics (actions, events, movement)
4. Environment, lighting, atmosphere
5. Camera work (angles, movements, transitions)

### Experiment 1A: Template Showdown

Test the same user prompt with different templates and compare outputs.

**User Prompt (keep constant):**
```
A woman walks through a neon-lit Tokyo street at night
```

**Test Matrix:**

| Template | Expected Emphasis |
|----------|-------------------|
| `none` (ComfyUI default) | Balanced, official behavior |
| `hunyuan_video_cinematic` | Dramatic lighting, dolly shots |
| `hunyuan_video_documentary` | Candid, handheld feel |
| `hunyuan_video_abstract` | Non-representational forms |
| `hunyuan_video_slowmo` | Motion details emphasized |
| Custom: Minimal | Just "Generate a video:" |

**What to observe:**
- Does camera movement differ?
- Is lighting interpreted differently?
- Does motion smoothness change?

### Experiment 1B: System Prompt Length

Test if verbose vs. minimal system prompts affect output quality.

**Minimal System Prompt:**
```
Generate a video.
```

**Medium System Prompt:**
```
You are a video generation assistant. Create a video based on the user's description,
paying attention to motion, camera movement, and visual quality.
```

**Verbose System Prompt:**
```
You are an expert cinematographer and video director. Analyze the user's prompt
carefully and generate a video that captures:
- The exact subject matter and setting described
- Natural, physics-respecting motion
- Professional camera work appropriate to the scene
- Consistent lighting throughout the sequence
- Smooth temporal transitions with no flickering
- High visual fidelity without artifacts
Consider the emotional tone and pacing. Use your expertise to fill in any details
not explicitly specified with cinematically appropriate choices.
```

**Hypothesis:** Medium-length focused prompts may outperform both minimal and verbose.

### Experiment 1C: Role-Based System Prompts

Test different "personas" for the system:

| Role | System Prompt |
|------|---------------|
| Director | "You are a film director creating a scene..." |
| Physicist | "You are a physics simulator generating accurate motion..." |
| Animator | "You are a 3D animator creating smooth keyframed motion..." |
| Poet | "You are a visual poet translating emotion into motion..." |

**User Prompt (same for all):**
```
Rain falling on a window
```

---

## Part 2: User Prompt Experiments

### The Anatomy of a Perfect Prompt

Based on analysis, the ideal structure is:

```
[Shot Type] + [Subject] + [Setting] + [Action] + [Camera] + [Lighting] + [Timing]
```

### Experiment 2A: Component Ablation

Test what happens when you remove each component:

**Full Prompt:**
```
A medium shot of a red cardinal perched on a snow-covered branch in a winter forest.
The bird tilts its head and then takes flight, wings spreading gracefully.
The camera holds steady, then follows the bird's ascent with a slow tilt upward.
Soft morning light filters through the trees. The entire motion takes 4 seconds.
```

**Remove Shot Type:**
```
A red cardinal perched on a snow-covered branch in a winter forest.
The bird tilts its head and then takes flight...
```

**Remove Camera Direction:**
```
A medium shot of a red cardinal perched on a snow-covered branch in a winter forest.
The bird tilts its head and then takes flight, wings spreading gracefully.
Soft morning light filters through the trees. The entire motion takes 4 seconds.
```

**Remove Timing:**
```
A medium shot of a red cardinal perched on a snow-covered branch in a winter forest.
The bird tilts its head and then takes flight, wings spreading gracefully.
The camera holds steady, then follows the bird's ascent with a slow tilt upward.
Soft morning light filters through the trees.
```

**Minimal Prompt:**
```
A bird takes flight from a branch
```

### Experiment 2B: Motion Vocabulary Test

Which motion verbs produce the best results?

**Test Set - Walking:**
```
A person walks through the forest
A person strolls through the forest
A person trudges through the forest
A person glides through the forest
A person marches through the forest
```

**Test Set - Water:**
```
Water flows down the river
Water rushes down the river
Water cascades down the river
Water trickles down the river
Water surges down the river
```

### Experiment 2C: Camera Movement Vocabulary

Test each camera term explicitly:

| Prompt | Expected Result |
|--------|-----------------|
| "The camera pans slowly to the right" | Horizontal rotation |
| "The camera tilts up toward the sky" | Vertical rotation |
| "The camera dollies forward toward the subject" | Push in |
| "The camera trucks left alongside the runner" | Parallel movement |
| "The camera orbits around the statue" | 360-degree circle |
| "The camera cranes up and over the crowd" | Elevated arc |
| "Handheld camera follows the action" | Intentional shake |
| "Smooth steadicam follows the dancer" | Stabilized tracking |

### Experiment 2D: Timing Precision

Does explicit timing improve temporal coherence?

**Vague Timing:**
```
A flower slowly blooms
```

**Moderate Timing:**
```
A flower blooms over several seconds
```

**Precise Timing:**
```
A flower blooms from closed bud to full bloom over exactly 4 seconds
```

**Frame-Based Timing:**
```
A flower blooms from closed bud to full bloom over 97 frames
```

### Experiment 2E: Multi-Subject Coordination

How well does the model handle multiple coordinated motions?

**Single Subject:**
```
A dancer spins gracefully
```

**Dual Coordinated:**
```
Two dancers spin in synchronized mirror movements, facing each other
```

**Opposing Motion:**
```
One dancer spins clockwise while her partner spins counterclockwise
```

**Complex Choreography:**
```
Three dancers: the center dancer leaps while the two flanking dancers
lower into graceful bows, all motions synchronized to complete together
```

---

## Part 3: Negative Prompt Experiments

### Experiment 3A: Negative Prompt Specificity

**Minimal Negative:**
```
low quality
```

**Standard Negative:**
```
low quality, blurry, distorted, artifacts, watermark, text, logo
```

**Motion-Focused Negative:**
```
low quality, blurry, distorted, static camera, no motion, jerky motion,
stuttering, flickering, temporal inconsistency, frame drops
```

**Comprehensive Negative:**
```
low quality, blurry, distorted, artifacts, watermark, text, logo,
static, no motion, jerky, stuttering, flickering, jittery,
morphing faces, distorted hands, unnatural movement, physics-breaking,
duplicate frames, temporal artifacts, compression artifacts
```

**Test:** Does a longer, more specific negative actually help?

### Experiment 3B: Contradictory Negatives

What happens with unusual negative prompts?

**Contradicting the positive:**
- Positive: "A sunny beach scene"
- Negative: "beach, sand, water, sun"

**Negating motion:**
- Positive: "A dancer performs an energetic routine"
- Negative: "motion, movement, dancing, energy"

**Negating the subject:**
- Positive: "A cat sleeps on a windowsill"
- Negative: "cat, animal, fur"

**Hypothesis:** Contradicting core elements may create interesting abstract results or complete failures.

### Experiment 3C: Style Negatives

Can you steer away from specific styles?

**Base Positive:** "A cityscape at night"

| Negative | Expected Effect |
|----------|-----------------|
| "anime, cartoon, illustration" | Push toward photorealism |
| "photorealistic, realistic, natural" | Push toward stylized |
| "dark, moody, noir" | Push toward bright/colorful |
| "colorful, saturated, vibrant" | Push toward muted/desaturated |

### Experiment 3D: Empty vs. Populated Negative

Test conditions:
1. Empty string negative (official default)
2. Simple negative: "low quality, blurry"
3. Detailed negative: "low quality, blurry, distorted, artifacts, watermark, flickering"

**Note:** Official HunyuanVideo-1.5 uses empty string as the default negative prompt.

---

## Part 4: CFG Scale Experiments

### Experiment 4A: CFG Sweep

Test CFG values: 1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0

**Scoring Criteria:**
- Prompt adherence (how closely does it match?)
- Visual quality (artifacts, sharpness)
- Motion smoothness (natural movement)
- Creativity (interesting choices not in prompt)

**Prompt for testing:**
```
A hummingbird hovers near a red flower, its wings a blur of motion.
The camera slowly pushes in. Golden afternoon light.
```

### Experiment 4B: CFG + Negative Strength

If using a setup that allows negative prompt weighting:

| Positive CFG | Negative Weight | Expected |
|--------------|-----------------|----------|
| 6.0 | 1.0 | Balanced |
| 6.0 | 0.5 | Less avoidance |
| 6.0 | 1.5 | Stronger avoidance |
| 8.0 | 1.0 | High adherence |
| 4.0 | 1.5 | Moderate + strong negative |

---

## Part 5: byT5 Multilingual Text Experiments

### How byT5 Activation Works

byT5 is triggered by quoted text:
- Double quotes: `"Hello World"`
- Chinese single quotes: `'Hello'`
- Chinese double quotes: `"Hello"`

The quoted content is extracted and encoded separately via byT5 for accurate text rendering.

### Experiment 5A: Text Rendering Accuracy

**Single Word:**
```
A neon sign that says "OPEN" glows in the window
```

**Multiple Words:**
```
A billboard displays "Welcome to Las Vegas" in bright lights
```

**Long Text:**
```
A movie theater marquee reads "Now Playing: The Great Adventure - Tickets Available"
```

**Special Characters:**
```
A digital clock displays "12:34" in red LED segments
```

### Experiment 5B: Multiple Quoted Strings

```
A storefront with two signs: "SALE" on the left and "50% OFF" on the right
```

**Question:** Does it render both correctly? In the right positions?

### Experiment 5C: Multilingual Text

**Chinese:**
```
A calligraphy scroll with the characters "和平" (peace) brushed elegantly
```

**Japanese:**
```
A Tokyo street sign reading "東京駅" illuminated at night
```

**Korean:**
```
A K-pop poster with "안녕하세요" prominently displayed
```

**Arabic:**
```
An ancient manuscript with "السلام" written in golden ink
```

### Experiment 5D: Mixed Language

```
A cosmopolitan cafe sign reading "Coffee カフェ Kaffee" in three scripts
```

### Experiment 5E: Quote Edge Cases

**Nested Quotes:**
```
A book cover with the title "He said 'Hello' to me"
```

**Escaped Characters:**
```
A math equation showing "E = mc²" on a chalkboard
```

**Empty Quotes:**
```
A speech bubble with "" appears above the character
```

---

## Part 6: Edge Case Experiments

### Experiment 6A: Contradictory Instructions

**Physics-Breaking:**
```
A ball falls upward into the sky while simultaneously rolling along the ground
```

**Temporal Paradox:**
```
The sun rises while setting, day and night coexisting in the same frame
```

**Self-Contradiction:**
```
A completely still object that moves rapidly across the screen
```

### Experiment 6B: Extreme Lengths

**One Word:**
```
Explosion
```

**Maximum Length (push limits):**
```
[Write a 500+ word detailed prompt with extensive descriptions of every element]
```

### Experiment 6C: Unusual Subjects

**Abstract Concept:**
```
The feeling of nostalgia visualized as flowing ribbons of warm color
```

**Impossible Object:**
```
A Penrose triangle slowly rotating in a white void
```

**Microscopic Scale:**
```
A cell dividing under a microscope, mitosis in progress
```

**Cosmic Scale:**
```
A galaxy slowly rotating over millions of years, compressed into 5 seconds
```

### Experiment 6D: Style Extremes

**Hyperrealism:**
```
Photorealistic 8K footage of a dewdrop sliding down a leaf, every microscopic detail visible
```

**Pure Abstraction:**
```
Non-representational forms of pure color morphing and flowing, no recognizable objects
```

**Retro/Vintage:**
```
Grainy 1920s silent film style footage of a woman in a flapper dress dancing
```

---

## Part 7: A/B Testing Protocols

### Quick Test Protocol

For rapid iteration:
1. Same seed for both variants
2. Same CFG, steps, resolution
3. Vary only the element being tested
4. Generate 2-3 samples each
5. Compare side-by-side

### Full Test Protocol

For thorough evaluation:
1. Generate 5 samples per variant (different seeds)
2. Score each on:
   - Prompt adherence (1-5)
   - Visual quality (1-5)
   - Motion quality (1-5)
   - Temporal consistency (1-5)
3. Calculate averages
4. Note any failure modes

### Scoring Rubric

**Prompt Adherence:**
- 5: All elements present and accurate
- 4: Most elements present, minor deviations
- 3: Main subject correct, secondary elements off
- 2: Partially related to prompt
- 1: Completely unrelated

**Visual Quality:**
- 5: No artifacts, sharp, professional
- 4: Minor artifacts, good overall
- 3: Noticeable artifacts but watchable
- 2: Significant quality issues
- 1: Unwatchable quality

**Motion Quality:**
- 5: Smooth, natural, physically accurate
- 4: Good motion with minor issues
- 3: Visible stuttering or unnatural movement
- 2: Jerky or inconsistent motion
- 1: No meaningful motion or broken

**Temporal Consistency:**
- 5: Perfect consistency frame-to-frame
- 4: Minor flickering or morphing
- 3: Noticeable inconsistencies
- 2: Significant changes between frames
- 1: Complete temporal incoherence

---

## Part 8: Fun Creative Experiments

### The Genre Gauntlet

Generate the same basic action in different genres:

**Base Action:** "A door opens"

| Genre | Prompt Modification |
|-------|---------------------|
| Horror | "A creaking door slowly opens in a dark abandoned mansion, shadows moving within" |
| Sci-Fi | "A futuristic airlock door slides open with a pneumatic hiss, revealing stars beyond" |
| Romance | "A door opens to reveal a long-lost lover standing in the rain" |
| Comedy | "A door opens and a pie flies out, splattering against the wall" |
| Documentary | "A weathered barn door swings open, revealing vintage tractors inside" |

### The Speed Run

Same subject at different speeds:

```
A cheetah running in slow motion, each muscle ripple visible
A cheetah running at normal speed across the savanna
A cheetah running in timelapse, a blur across the landscape
```

### The Camera Challenge

Same scene, every camera move:

**Scene:** "A single tree in an open field"

1. Static wide shot
2. Slow push in
3. Slow pull out
4. Pan around the tree
5. Tilt from roots to canopy
6. Orbit 360 degrees
7. Crane shot rising up
8. Drone shot diving down
9. Handheld approach
10. Rack focus from grass to tree

### The Impossible Shot

Test the model's limits with cinematically impossible requests:

```
A single continuous shot that starts underwater, rises through the surface,
follows a seagull into the sky, then dives back down into a forest canopy
```

```
A bullet-time freeze frame that slowly rotates 360 degrees around a
dancer frozen mid-leap, then time resumes and she lands gracefully
```

---

## Part 9: Results Documentation Template

For each experiment, document:

```markdown
## Experiment [Number]: [Name]

**Date:** YYYY-MM-DD
**Template Used:** [template name or "none"]
**CFG Scale:** X.X
**Steps:** XX
**Resolution:** WxH
**Frames:** XX

### Variants Tested

| Variant | Prompt | Score (Adherence/Quality/Motion/Temporal) |
|---------|--------|-------------------------------------------|
| A | ... | X/X/X/X |
| B | ... | X/X/X/X |

### Observations

[What differences did you notice?]

### Conclusions

[What did you learn?]

### Recommended Follow-up

[What should be tested next?]
```

---

## Quick Reference: Prompt Patterns That Work

### Camera Movements
- "The camera [slowly/quickly] [pans/tilts/dollies/trucks/orbits] [direction]"
- "Smooth steadicam follows..."
- "Handheld camera tracks..."
- "Static wide shot holds on..."

### Motion Descriptions
- "[Subject] [verb]s [adverb] over [X] seconds"
- "Beginning from [state A], transitioning to [state B]"
- "[Action A] while simultaneously [Action B]"

### Lighting
- "Golden hour lighting"
- "Dramatic chiaroscuro shadows"
- "Soft diffused overcast light"
- "Harsh noon sun"
- "Neon-lit night scene"

### Atmosphere
- "Peaceful and serene"
- "Tense and suspenseful"
- "Dreamlike and ethereal"
- "Gritty and realistic"

### Timing
- "over X seconds"
- "in slow motion"
- "gradually"
- "suddenly"
- "completing at the X second mark"

---

## Appendix: Suggested Negative Prompts

The official HunyuanVideo-1.5 uses an **empty string** as the default negative prompt.
You can experiment with adding negative prompts for potentially better results:

**Minimal:**
```
low quality, blurry
```

**Standard:**
```
low quality, blurry, distorted, artifacts, watermark, text, logo
```

**Motion-focused:**
```
low quality, blurry, distorted, artifacts, watermark, text, logo,
static camera, no motion, jerky motion, stuttering, flickering
```

Experiment to find what works best for your use case.
