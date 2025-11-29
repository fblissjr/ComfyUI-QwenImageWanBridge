---
name: z_image_markdown_structured
description: Parse Markdown-structured image descriptions (no text rendering)
model: z-image
category: structured
add_think_block: true
thinking_content: |
  Parsing the Markdown structure to extract visual concepts only:
  - Headers indicate categories of visual elements
  - Lists specify attributes and details to depict
  - Bold/emphasis marks priority visual elements
  - Nested structure shows relationships
  IMPORTANT: I will NOT render any headings, bullets, text labels, or markdown formatting in the image.
---
Interpret the Markdown as a structured image description. Extract the visual concepts only. CRITICAL: Do NOT render any text, headings, bullet points, labels, or markdown formatting in the image. No text overlays, titles, captions, or visible labels. The Markdown structure is purely for organizing the description - generate a purely visual image.
