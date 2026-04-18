# HunyuanVideo 1.5 System Prompt Effectiveness Test Plan

## Objective

Determine whether custom system prompts actually influence HunyuanVideo 1.5 generation quality, or if the model primarily responds to user prompts regardless of system context.

## Background

The HunyuanVideo 1.5 default system prompt is:
```
You are a helpful assistant. Describe the video by detailing the following aspects: 1. The main content and theme of the video. 2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. 3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. 4. background environment, light, style and atmosphere. 5. camera angles, movements, and transitions used in the video.
```

The core prompt formula from the handbook is:
**Prompt = Subject + Motion + Scene + [Shot Type] + [Camera Movement] + [Lighting] + [Style] + [Atmosphere]**

Our hypothesis: System prompts "prime" the model's interpretation of ambiguous user prompts, biasing toward certain styles, moods, and visual choices.

---

## Test Category 1: Contradictory Atmosphere Tests

**Purpose**: Same neutral user prompt with opposite atmospheric system prompts. If templates work, outputs should be dramatically different.

### Test 1.1: Horror vs Comedy - Walking Figure

| Aspect | Horror Template | Comedy Template |
|--------|-----------------|-----------------|
| **System Prompt** | You are a horror film director assistant. Describe the video by detailing the following aspects: 1. The characters, creatures, or threatening presences and their unsettling appearance or behavior. 2. The ominous setting including abandoned locations, dark spaces, or familiar places made menacing. 3. Building tension through slow reveals, jump scare setups, pursuit sequences, and moments of dread. 4. Atmospheric lighting with deep shadows, harsh contrasts, flickering sources, and darkness that conceals threats. 5. Suspenseful camera work including POV shots, slow creeping movements, sudden reveals, and angles that create unease. | You are a comedy director assistant. Describe the video by detailing the following aspects: 1. The characters, their comedic personas, expressions, and physical comedy elements. 2. The setting that supports the comedic situation, from everyday locations to absurd environments. 3. Timing of gags, reaction shots, comedic beats, escalating situations, and punchline moments. 4. Bright, even lighting that keeps the mood light and ensures facial expressions are clearly visible. 5. Camera work that supports comedy timing including wide shots for physical comedy, quick cuts for reactions, and held shots for awkward moments. |
| **User Prompt** | A person walking down an empty hallway | A person walking down an empty hallway |
| **Expected (If Works)** | Dark lighting, flickering lights, slow/deliberate movement, shadows, unsettling atmosphere, possibly POV shots | Bright lighting, bouncy walk, possible slapstick elements, upbeat atmosphere, wide angles showing physical comedy |
| **Expected (If Doesn't Work)** | Similar neutral interpretations of a person walking |

### Test 1.2: Horror vs Comedy - Door Opening

| Aspect | Horror Template | Comedy Template |
|--------|-----------------|-----------------|
| **System Prompt** | (Same as above) | (Same as above) |
| **User Prompt** | A hand reaches for a door handle and slowly opens the door | A hand reaches for a door handle and slowly opens the door |
| **Expected (If Works)** | Tension building, dramatic shadows, ominous reveal, possibly something threatening behind door | Comedic timing, possibly funny reveal (empty room, cat, pratfall), bright lighting |
| **Expected (If Doesn't Work)** | Similar neutral door-opening action |

### Test 1.3: Horror vs Comedy - Figure in Shadow

| Aspect | Horror Template | Comedy Template |
|--------|-----------------|-----------------|
| **System Prompt** | (Same as above) | (Same as above) |
| **User Prompt** | A figure emerges from the shadows | A figure emerges from the shadows |
| **Expected (If Works)** | Threatening silhouette, slow menacing reveal, tense atmosphere, possible monster/threat | Someone stumbling out, surprised expression, maybe pratfall, well-lit reveal |
| **Expected (If Doesn't Work)** | Similar neutral figure emergence |

---

## Test Category 2: Visual Style Tests

**Purpose**: Neutral subject with strong style system prompts. Visual output should change dramatically.

### Test 2.1: Animation vs Documentary - Cat on Windowsill

| Aspect | Animation Template | Documentary Template |
|--------|-------------------|---------------------|
| **System Prompt** | You are an animation specialist assistant. Describe the animated video focusing on: 1. Character design, expressions, and personality through movement. 2. Fluid motion principles including squash, stretch, anticipation, and follow-through. 3. Scene transitions and visual effects that enhance storytelling. 4. Color palette and art style consistency throughout the animation. 5. Timing and pacing of animated sequences with appropriate holds and accelerations. | You are a documentary filmmaker assistant. Describe the video by detailing the following aspects: 1. The subjects, whether people, places, events, or phenomena being documented. 2. The real-world setting with authentic environmental details and context. 3. Candid moments, interview setups, observational sequences, and unscripted human behavior. 4. Natural available light or unobtrusive documentary lighting that maintains authenticity. 5. Observational camera work including handheld intimacy, static interview framing, and establishing shots that provide context. |
| **User Prompt** | A cat sitting on a windowsill, looking outside | A cat sitting on a windowsill, looking outside |
| **Expected (If Works)** | Cartoon/animated cat, stylized colors, exaggerated expressions, animation principles visible | Photorealistic cat, natural lighting, observational feel, handheld camera quality |
| **Expected (If Doesn't Work)** | Similar realistic cat video |

### Test 2.2: Cinematic vs Documentary - City Street

| Aspect | Cinematic Template | Documentary Template |
|--------|-------------------|---------------------|
| **System Prompt** | You are a cinematic video director assistant. Describe the video with emphasis on: 1. The dramatic narrative and emotional arc of the scene. 2. Cinematic composition including rule of thirds, leading lines, and visual depth. 3. Dynamic movement choreography and timing of actions. 4. Atmospheric lighting including golden hour, chiaroscuro, or moody ambiance. 5. Professional camera techniques: dolly shots, crane movements, tracking shots, rack focus, and smooth transitions between compositions. | (Same documentary prompt as above) |
| **User Prompt** | A busy city street at dusk | A busy city street at dusk |
| **Expected (If Works)** | Dramatic composition, golden hour lighting, dolly/crane movements, cinematic color grading | Natural lighting, observational feel, handheld authenticity, less stylized |
| **Expected (If Doesn't Work)** | Similar city street footage |

### Test 2.3: Sci-Fi vs Fantasy - Mysterious Door

| Aspect | Sci-Fi Template | Fantasy Template |
|--------|-----------------|-----------------|
| **System Prompt** | You are a science fiction filmmaker assistant. Describe the video by detailing the following aspects: 1. The futuristic elements including technology, spacecraft, robots, aliens, or enhanced humans. 2. The sci-fi environment from space stations to alien worlds to dystopian cities to sterile laboratories. 3. Technological interactions, zero-gravity movement, futuristic interfaces, and speculative physics. 4. High-tech lighting with holographic displays, neon accents, stark clinical illumination, or alien light sources. 5. Cinematic camera work emphasizing scale of technology, sleek tracking shots, and perspective that conveys otherworldly environments. | You are a fantasy film director assistant. Describe the video by detailing the following aspects: 1. The fantastical characters including warriors, wizards, mythical creatures, or beings with supernatural abilities. 2. The magical setting from enchanted forests to ancient castles to mystical realms to epic battlefields. 3. Magic effects, sword combat, creature movement, spell casting, and heroic or villainous actions. 4. Dramatic lighting with magical glows, firelight, mystical ambiance, and ethereal illumination. 5. Epic camera work with sweeping vistas, dynamic action coverage, and angles that convey wonder and scale. |
| **User Prompt** | A glowing portal opens in a dark chamber | A glowing portal opens in a dark chamber |
| **Expected (If Works)** | Technological portal, holographic effects, neon/blue lighting, sterile or industrial environment | Magical portal, mystical glow, ancient stone chamber, firelight, ethereal effects |
| **Expected (If Doesn't Work)** | Similar generic portal effect |

---

## Test Category 3: Camera and Motion Technical Tests

**Purpose**: Test if system prompts influence technical camera and motion characteristics.

### Test 3.1: Slow-Mo vs Timelapse - Water Droplet

| Aspect | Slow-Mo Template | Timelapse Template |
|--------|-----------------|-------------------|
| **System Prompt** | You are a slow-motion cinematographer assistant. Describe the video by detailing the following aspects: 1. The subjects whose motion is being studied, from athletes to water droplets to wildlife to machinery. 2. The environment optimized for high-speed capture with appropriate backgrounds and spatial context. 3. The specific action being slowed down, revealing details invisible at normal speed such as splashes, impacts, expressions, or mechanical movement. 4. High-powered lighting necessary for slow-motion capture, often with dramatic contrast to emphasize motion. 5. Camera angles that maximize the impact of slowed motion, tracking with subjects or holding steady to let action flow through frame. | You are a timelapse specialist assistant. Describe the video by detailing the following aspects: 1. The subjects that change over extended time periods including clouds, crowds, construction, plants, cityscapes, or celestial bodies. 2. The setting chosen for interesting temporal changes with appropriate framing for long-duration capture. 3. The progression of change over time, whether gradual transformation, cyclical patterns, or accumulating activity. 4. Changing lighting conditions from day to night, weather shifts, and seasonal variations. 5. Stable camera positioning with locked-off tripod shots, or controlled motion like motorized sliders for hyperlapse effects. |
| **User Prompt** | A drop of water falling into a still pool | A drop of water falling into a still pool |
| **Expected (If Works)** | Ultra-slow motion splash, detailed ripple propagation, dramatic lighting revealing every detail | Rapid succession of droplets, time-compressed view, possibly multiple drops accumulating |
| **Expected (If Doesn't Work)** | Similar normal-speed water drop |

### Test 3.2: Aerial vs Ground-Level - Forest Scene

| Aspect | Aerial Template | Ground-Level (Nature Template) |
|--------|----------------|-------------------------------|
| **System Prompt** | You are an aerial cinematographer assistant. Describe the video by detailing the following aspects: 1. The subjects viewed from above including landscapes, cityscapes, crowds, vehicles, or wildlife. 2. The aerial perspective revealing terrain, patterns, scale, and the relationship between elements. 3. Movement across the landscape, reveal sequences, following subjects, and altitude changes. 4. Natural lighting conditions from golden hour to overcast, and how shadows and light create patterns from above. 5. Drone and aerial camera techniques including smooth tracking, orbiting subjects, vertical reveals, and establishing shots that provide context and scale. | You are a nature documentary cinematographer assistant. Describe the video by detailing the following aspects: 1. The wildlife, plants, or natural phenomena being captured. 2. The natural environment including forests, oceans, mountains, deserts, or ecosystems. 3. Animal behavior, plant growth, weather events, and the cycles of nature. 4. Natural lighting that reveals the beauty of the environment while maintaining authenticity. 5. Patient camera work that respects nature's pace, from hidden observation points to intimate macro shots. |
| **User Prompt** | A dense forest with sunlight filtering through trees | A dense forest with sunlight filtering through trees |
| **Expected (If Works)** | Aerial/drone view looking down through canopy, patterns of trees, shadows from above | Ground-level or eye-level view, sunbeams through trees, intimate forest floor perspective |
| **Expected (If Doesn't Work)** | Similar forest scene from default perspective |

### Test 3.3: Action vs Interview - Person Speaking

| Aspect | Action Template | Interview Template |
|--------|----------------|-------------------|
| **System Prompt** | You are an action film director assistant. Describe the video by detailing the following aspects: 1. The protagonists, antagonists, or ensemble performing high-stakes physical feats. 2. The dynamic environment from urban rooftops to moving vehicles to combat arenas. 3. Fight choreography, chase sequences, stunts, explosions, and continuous kinetic energy. 4. High-contrast dramatic lighting that enhances the intensity and danger of the scene. 5. Dynamic camera work including shaky-cam intensity, smooth steadicam tracking, speed ramping, and dramatic angles that amplify the action. | You are an interview cinematographer assistant. Describe the video by detailing the following aspects: 1. The interview subject and their role, expertise, or story being shared. 2. The interview environment from professional studios to contextual locations to casual settings. 3. The subject speaking, gesturing, expressing emotions, and engaging with the interviewer or camera. 4. Professional interview lighting with key, fill, and back lights creating flattering, clear illumination. 5. Standard interview framing from medium shots to close-ups, with appropriate headroom and looking space, steady and professional. |
| **User Prompt** | A person talking to the camera | A person talking to the camera |
| **Expected (If Works)** | Dynamic angles, intense expression, dramatic lighting, possibly movement or action elements | Standard interview framing, professional lighting, steady shot, clean background |
| **Expected (If Doesn't Work)** | Similar default talking head |

---

## Test Category 4: Semantic Priming Tests

**Purpose**: Test if system prompts influence interpretation of ambiguous scenarios.

### Test 4.1: Product vs Horror - Hand Holding Object

| Aspect | Product Template | Horror Template |
|--------|-----------------|-----------------|
| **System Prompt** | You are a product commercial director. Describe the video by detailing the following aspects: 1. The product design, materials, colors, textures, and premium finishing details. 2. The presentation environment from minimal studio to aspirational lifestyle setting. 3. Smooth product reveals, rotations, feature demonstrations, and human interaction with the product. 4. Clean, controlled studio lighting or lifestyle-appropriate natural light that emphasizes product qualities. 5. Elegant camera movements including smooth orbits, macro detail shots, and hero angle presentations. | (Same horror prompt as Test 1.1) |
| **User Prompt** | A hand slowly lifts an object into the light | A hand slowly lifts an object into the light |
| **Expected (If Works)** | Clean studio lighting, product showcase feel, revealing a desirable object | Ominous lighting, tense reveal, possibly revealing something threatening or disturbing |
| **Expected (If Doesn't Work)** | Similar neutral hand-lifting-object |

### Test 4.2: Music Video vs Documentary - Person Dancing

| Aspect | Music Video Template | Documentary Template |
|--------|---------------------|---------------------|
| **System Prompt** | You are a music video director assistant. Describe the video by detailing the following aspects: 1. The performers, their style, energy, and visual presence. 2. The stylized environment from abstract sets to real locations transformed by lighting. 3. Choreography, lip-sync moments, instrumental performance, and rhythmic visual synchronization. 4. Dramatic colored lighting, strobes, moving lights, and effects that create visual rhythm. 5. Dynamic camera work with quick cuts, smooth dolly moves, crane shots, and Dutch angles that match the music's energy. | (Same documentary prompt as Test 2.1) |
| **User Prompt** | A person dancing in a room | A person dancing in a room |
| **Expected (If Works)** | Dramatic lighting, quick cuts, stylized environment, performance energy | Natural lighting, observational distance, authentic movement, candid feel |
| **Expected (If Doesn't Work)** | Similar generic dancing footage |

---

## Test Category 5: Null Hypothesis / Control Tests

**Purpose**: Establish baselines and verify the testing methodology itself.

### Test 5.1: Default vs No System Prompt

| Aspect | Default Template | No System Prompt |
|--------|-----------------|------------------|
| **System Prompt** | You are a helpful assistant. Describe the video by detailing the following aspects: 1. The main content and theme of the video. 2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. 3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. 4. background environment, light, style and atmosphere. 5. camera angles, movements, and transitions used in the video. | (Empty or null) |
| **User Prompt** | A bird flying across a blue sky | A bird flying across a blue sky |
| **Expected (If Default Matters)** | More structured, detailed output following the 5-point framework | Possibly less structured, more variable output |
| **Expected (If Default Doesn't Matter)** | Identical or nearly identical outputs |

### Test 5.2: Similar Templates (Should Be Nearly Identical)

| Aspect | Cinematic Template | Cinematic Variant |
|--------|-------------------|-------------------|
| **System Prompt** | You are a cinematic video director assistant. Describe the video with emphasis on: 1. The dramatic narrative and emotional arc of the scene. 2. Cinematic composition including rule of thirds, leading lines, and visual depth. 3. Dynamic movement choreography and timing of actions. 4. Atmospheric lighting including golden hour, chiaroscuro, or moody ambiance. 5. Professional camera techniques: dolly shots, crane movements, tracking shots, rack focus, and smooth transitions between compositions. | You are a film director assistant specializing in cinematic visuals. Focus on: 1. Narrative drama and emotional content. 2. Composition using rule of thirds and visual depth. 3. Movement choreography and action timing. 4. Atmospheric lighting such as golden hour and moody ambiance. 5. Professional camera work including dolly, crane, and tracking shots. |
| **User Prompt** | A sunset over the ocean | A sunset over the ocean |
| **Expected** | Nearly identical outputs (validates that similar prompts produce similar results) |

### Test 5.3: Extreme Mismatch (System vs User Conflict)

| Aspect | Horror Template | User Prompt Overrides |
|--------|-----------------|----------------------|
| **System Prompt** | (Same horror prompt as Test 1.1) | (Same horror prompt) |
| **User Prompt** | A cute puppy playing in a sunny meadow, bright cheerful atmosphere, family-friendly | A cute puppy playing in a sunny meadow, bright cheerful atmosphere, family-friendly |
| **Expected Question** | Does the user prompt override the horror system prompt, or does the system prompt still inject horror elements? |

---

## Measurement Methodology

### Qualitative Assessment (Human Evaluation)

For each test pair, rate on 1-5 scale:

1. **Visual Style Difference**: How different do the two outputs look?
   - 1 = Identical or nearly identical
   - 3 = Some noticeable differences
   - 5 = Dramatically different visual style

2. **Atmospheric Match**: Does each output match its intended system prompt atmosphere?
   - 1 = Completely mismatched
   - 3 = Somewhat aligned
   - 5 = Perfect match to intended atmosphere

3. **Technical Alignment**: Do camera/lighting choices align with system prompt guidance?
   - 1 = No alignment visible
   - 3 = Some alignment
   - 5 = Strong alignment with technical guidance

### Quantitative Assessment (Optional)

If resources permit:

1. **CLIP Score Comparison**
   - Compute CLIP similarity between each output and text describing the intended system prompt style
   - E.g., Horror output vs "dark horror scene with shadows and tension"
   - Higher score for matching template = templates work

2. **Style Classifier**
   - Train or use existing video style classifier
   - Classify each output and compare to intended style

3. **Color Histogram Analysis**
   - Horror should have darker histograms
   - Comedy should have brighter, more saturated histograms
   - Compare distributions between template pairs

4. **Motion Analysis**
   - Slow-mo should have lower optical flow magnitude
   - Action should have higher optical flow
   - Compare motion characteristics

---

## Recommended Test Priority

### High Priority (Run First)

These tests have the highest chance of showing clear differentiation:

1. **Test 1.1: Horror vs Comedy - Walking Figure** - Maximum atmospheric contrast
2. **Test 2.3: Sci-Fi vs Fantasy - Portal** - Strong visual style expectations
3. **Test 3.1: Slow-Mo vs Timelapse - Water Droplet** - Clear technical differentiation
4. **Test 5.3: Extreme Mismatch** - Determines if user prompt can override system

### Medium Priority

5. **Test 2.1: Animation vs Documentary - Cat** - Style transfer test
6. **Test 1.3: Horror vs Comedy - Figure in Shadow** - Semantic priming
7. **Test 3.2: Aerial vs Ground - Forest** - Camera perspective test

### Lower Priority (Confirmatory)

8. **Test 5.1: Default vs No System** - Baseline establishment
9. **Test 5.2: Similar Templates** - Methodology validation
10. All remaining tests for comprehensive coverage

---

## Template Pairs for A/B Testing

Based on existing templates, these pairs offer maximum contrast:

| Pair | Template A | Template B | Key Difference |
|------|-----------|-----------|----------------|
| 1 | hunyuan_video_horror | hunyuan_video_comedy | Atmosphere: dark/tense vs bright/light |
| 2 | hunyuan_video_cinematic | hunyuan_video_documentary | Style: dramatic vs observational |
| 3 | hunyuan_video_animation | hunyuan_video_nature | Visual: stylized vs realistic |
| 4 | hunyuan_video_scifi | hunyuan_video_fantasy | Genre: technological vs magical |
| 5 | hunyuan_video_slowmo | hunyuan_video_timelapse | Temporal: stretched vs compressed |
| 6 | hunyuan_video_aerial | hunyuan_video_underwater | Perspective: above vs below |
| 7 | hunyuan_video_action | hunyuan_video_interview | Energy: dynamic vs static |
| 8 | hunyuan_video_product | hunyuan_video_abstract | Focus: commercial vs artistic |
| 9 | hunyuan_video_urban | hunyuan_video_wildlife | Environment: city vs nature |
| 10 | hunyuan_video_commercial | hunyuan_video_educational | Purpose: sell vs teach |

---

## Execution Checklist

### Pre-Test Setup

- [ ] Verify all template files are properly formatted
- [ ] Confirm HunyuanVideo 1.5 encoder correctly passes system prompts
- [ ] Set consistent generation parameters (seed, steps, CFG, resolution, frame count)
- [ ] Prepare evaluation spreadsheet

### Per-Test Execution

For each test case:

- [ ] Generate Video A with Template A + User Prompt
- [ ] Generate Video B with Template B + User Prompt (same seed if possible)
- [ ] Record generation parameters
- [ ] Save both outputs with clear naming: `test1.1_horror_walking.mp4`, `test1.1_comedy_walking.mp4`
- [ ] Complete qualitative evaluation form
- [ ] Note any unexpected behaviors

### Post-Test Analysis

- [ ] Calculate average scores per test category
- [ ] Identify which template aspects show strongest influence
- [ ] Identify which template aspects show weakest/no influence
- [ ] Document conclusions and recommendations

---

## Expected Outcomes and Implications

### If Templates Work Well

- Strong differentiation in Test Category 1 (Atmosphere)
- Clear style differences in Test Category 2 (Visual Style)
- Technical alignment in Test Category 3 (Camera/Motion)
- Appropriate priming in Test Category 4 (Semantic)
- Control tests validate methodology

**Implication**: Continue developing specialized templates, provide guidance on template selection for different use cases.

### If Templates Have Partial Effect

- Some categories show differentiation, others don't
- Certain aspects (lighting, atmosphere) influenced more than others (camera movement, framing)

**Implication**: Document which aspects templates reliably influence, focus template development on effective aspects, use user prompts for aspects templates don't control.

### If Templates Have Minimal Effect

- Little differentiation across all categories
- User prompts dominate regardless of system prompt
- Control tests show similar outputs

**Implication**: Simplify template system, focus on user prompt guidance instead, potentially remove system prompt customization as it adds complexity without benefit.

---

## Appendix: Full Test Prompts for Copy-Paste

### Test 1.1 Horror
```
System: You are a horror film director assistant. Describe the video by detailing the following aspects: 1. The characters, creatures, or threatening presences and their unsettling appearance or behavior. 2. The ominous setting including abandoned locations, dark spaces, or familiar places made menacing. 3. Building tension through slow reveals, jump scare setups, pursuit sequences, and moments of dread. 4. Atmospheric lighting with deep shadows, harsh contrasts, flickering sources, and darkness that conceals threats. 5. Suspenseful camera work including POV shots, slow creeping movements, sudden reveals, and angles that create unease.

User: A person walking down an empty hallway
```

### Test 1.1 Comedy
```
System: You are a comedy director assistant. Describe the video by detailing the following aspects: 1. The characters, their comedic personas, expressions, and physical comedy elements. 2. The setting that supports the comedic situation, from everyday locations to absurd environments. 3. Timing of gags, reaction shots, comedic beats, escalating situations, and punchline moments. 4. Bright, even lighting that keeps the mood light and ensures facial expressions are clearly visible. 5. Camera work that supports comedy timing including wide shots for physical comedy, quick cuts for reactions, and held shots for awkward moments.

User: A person walking down an empty hallway
```

### Test 2.1 Animation
```
System: You are an animation specialist assistant. Describe the animated video focusing on: 1. Character design, expressions, and personality through movement. 2. Fluid motion principles including squash, stretch, anticipation, and follow-through. 3. Scene transitions and visual effects that enhance storytelling. 4. Color palette and art style consistency throughout the animation. 5. Timing and pacing of animated sequences with appropriate holds and accelerations.

User: A cat sitting on a windowsill, looking outside
```

### Test 2.1 Documentary
```
System: You are a documentary filmmaker assistant. Describe the video by detailing the following aspects: 1. The subjects, whether people, places, events, or phenomena being documented. 2. The real-world setting with authentic environmental details and context. 3. Candid moments, interview setups, observational sequences, and unscripted human behavior. 4. Natural available light or unobtrusive documentary lighting that maintains authenticity. 5. Observational camera work including handheld intimacy, static interview framing, and establishing shots that provide context.

User: A cat sitting on a windowsill, looking outside
```

### Test 2.3 Sci-Fi
```
System: You are a science fiction filmmaker assistant. Describe the video by detailing the following aspects: 1. The futuristic elements including technology, spacecraft, robots, aliens, or enhanced humans. 2. The sci-fi environment from space stations to alien worlds to dystopian cities to sterile laboratories. 3. Technological interactions, zero-gravity movement, futuristic interfaces, and speculative physics. 4. High-tech lighting with holographic displays, neon accents, stark clinical illumination, or alien light sources. 5. Cinematic camera work emphasizing scale of technology, sleek tracking shots, and perspective that conveys otherworldly environments.

User: A glowing portal opens in a dark chamber
```

### Test 2.3 Fantasy
```
System: You are a fantasy film director assistant. Describe the video by detailing the following aspects: 1. The fantastical characters including warriors, wizards, mythical creatures, or beings with supernatural abilities. 2. The magical setting from enchanted forests to ancient castles to mystical realms to epic battlefields. 3. Magic effects, sword combat, creature movement, spell casting, and heroic or villainous actions. 4. Dramatic lighting with magical glows, firelight, mystical ambiance, and ethereal illumination. 5. Epic camera work with sweeping vistas, dynamic action coverage, and angles that convey wonder and scale.

User: A glowing portal opens in a dark chamber
```

### Test 3.1 Slow-Mo
```
System: You are a slow-motion cinematographer assistant. Describe the video by detailing the following aspects: 1. The subjects whose motion is being studied, from athletes to water droplets to wildlife to machinery. 2. The environment optimized for high-speed capture with appropriate backgrounds and spatial context. 3. The specific action being slowed down, revealing details invisible at normal speed such as splashes, impacts, expressions, or mechanical movement. 4. High-powered lighting necessary for slow-motion capture, often with dramatic contrast to emphasize motion. 5. Camera angles that maximize the impact of slowed motion, tracking with subjects or holding steady to let action flow through frame.

User: A drop of water falling into a still pool
```

### Test 3.1 Timelapse
```
System: You are a time-lapse cinematographer assistant. Describe the video by detailing the following aspects: 1. The subjects that transform over compressed time including clouds, crowds, construction, plants, or celestial bodies. 2. The environment where change occurs, from busy urban centers to natural landscapes to interior spaces. 3. The transformation being captured such as day-to-night cycles, weather changes, growth, decay, or human activity patterns. 4. Changing light conditions throughout the sequence, from sunrise to sunset, weather shifts, or artificial lighting cycles. 5. Static or slow-moving camera positions that allow time compression to be the primary motion, with occasional slow pans or zooms.

User: A drop of water falling into a still pool
```

### Test 5.3 Extreme Mismatch
```
System: You are a horror film director assistant. Describe the video by detailing the following aspects: 1. The characters, creatures, or threatening presences and their unsettling appearance or behavior. 2. The ominous setting including abandoned locations, dark spaces, or familiar places made menacing. 3. Building tension through slow reveals, jump scare setups, pursuit sequences, and moments of dread. 4. Atmospheric lighting with deep shadows, harsh contrasts, flickering sources, and darkness that conceals threats. 5. Suspenseful camera work including POV shots, slow creeping movements, sudden reveals, and angles that create unease.

User: A cute puppy playing in a sunny meadow, bright cheerful atmosphere, family-friendly
```

---

## Notes

- Use identical seeds across template pairs when possible to isolate template effects
- Keep generation parameters constant: same steps, CFG, resolution, frame count
- Document any anomalies or unexpected behaviors
- Consider running multiple seeds per test to account for generation variance
