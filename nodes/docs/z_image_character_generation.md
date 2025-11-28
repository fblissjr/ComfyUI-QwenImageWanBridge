# Z-Image Character Generation Guide

Using `thinking_content` to provide structured character specifications.

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
