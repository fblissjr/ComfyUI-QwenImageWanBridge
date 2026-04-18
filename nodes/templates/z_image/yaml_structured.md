---
name: z_image_yaml_structured
description: Parse YAML-structured image descriptions (no text rendering)
model: z-image
category: structured
add_think_block: true
thinking_content: |
  Parsing the YAML hierarchy to extract visual concepts only:
  - Core subject and composition (what to depict visually)
  - Style attributes and references (how it should look)
  - Environmental and lighting details (mood and tone)
  - Nested relationships between elements (arrangement)
  IMPORTANT: I will NOT render any text, labels, keys, or YAML formatting in the image.
---
Interpret the YAML as a structured image description. Extract the visual concepts from the values only. CRITICAL: Do NOT render any text, labels, keys, colons, or structured formatting in the image. No text overlays, titles, headings, or captions. The YAML hierarchy is purely for organizing the description - generate a purely visual image.
