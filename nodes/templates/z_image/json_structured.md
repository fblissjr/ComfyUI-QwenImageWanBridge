---
name: z_image_json_structured
description: Parse JSON-structured image descriptions (no text rendering)
model: z-image
category: structured
add_think_block: true
thinking_content: |
  Parsing the JSON structure to extract visual concepts only:
  - Subject and scene elements (what to depict visually)
  - Style and artistic direction (how it should look)
  - Lighting and atmosphere (mood and tone)
  - Composition and framing (arrangement)
  IMPORTANT: I will NOT render any text, labels, keys, or JSON formatting in the image.
---
Interpret the JSON as a structured image description. Extract the visual concepts from the values only. CRITICAL: Do NOT render any text, labels, keys, quotation marks, or structured formatting in the image. No text overlays, titles, headings, or captions. The JSON structure is purely for organizing the description - generate a purely visual image.
