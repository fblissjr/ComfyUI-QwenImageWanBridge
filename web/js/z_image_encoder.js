/**
 * Z-Image Text Encoder UI Extension
 * Auto-fills system_prompt when template_preset changes
 *
 * NOTE: Templates are loaded from nodes/templates/z_image_*.md files by Python.
 * This JS file contains a subset for auto-fill. Templates not listed here
 * still work - you can edit system_prompt manually for any template.
 */

import { app } from "../../../scripts/app.js";

// Common Z-Image templates for auto-fill
// Full list loaded from nodes/templates/z_image_*.md by Python
const Z_IMAGE_TEMPLATES = {
  none: "",
  default: "",
  photorealistic: "Generate a photorealistic image with accurate lighting, natural textures, and realistic details. Pay attention to proper shadows, reflections, and material properties.",
  artistic: "Generate an artistic, aesthetically pleasing image with creative composition, harmonious colors, and visual balance. Focus on artistic expression and visual impact.",
  bilingual_text: "Generate an image with accurate text rendering. Support both English and Chinese text with clear, readable typography.",
  // Photography styles
  portrait_studio: "Professional studio portrait photography. Clean background, controlled lighting with key and fill lights, sharp focus on subject, professional quality headshot or portrait composition.",
  portrait_environmental: "Environmental portrait showing subject in their natural setting. Context-rich composition, natural lighting preferred, tells a story about the subject through their surroundings.",
  landscape_epic: "Epic landscape photography. Dramatic scale and depth, powerful natural lighting, wide angle composition showing grandeur, sharp focus from foreground to background.",
  landscape_intimate: "Intimate landscape focusing on smaller details and moments. Soft lighting, thoughtful composition highlighting textures and patterns in nature.",
  // Art styles
  oil_painting_classical: "Classical oil painting style with rich colors, visible brushwork, and masterful light handling. Renaissance or Baroque influence with dramatic chiaroscuro.",
  oil_painting_impressionist: "Impressionist oil painting with visible brushstrokes, emphasis on light and color over detail, capturing fleeting moments and atmospheric effects.",
  watercolor_loose: "Loose, expressive watercolor style. Wet-on-wet techniques, soft edges, beautiful color bleeds and blooms, spontaneous and fluid feeling.",
  digital_concept: "Professional digital concept art. Clean rendering, strong silhouettes, clear focal points, industry-standard quality suitable for game or film pre-production.",
  // Lighting
  golden_hour: "Golden hour lighting with warm, soft directional light from low sun angle. Long shadows, rich warm tones, magical atmospheric quality.",
  blue_hour: "Blue hour twilight lighting. Cool ambient tones, city lights beginning to glow, peaceful transition between day and night.",
  studio_dramatic: "Dramatic studio lighting with strong contrast. Single key light creating bold shadows, high contrast ratio, theatrical and impactful.",
  neon_cyberpunk: "Neon-lit cyberpunk atmosphere. Multiple colored light sources, strong color contrast, reflective wet surfaces, urban dystopian mood.",
  noir: "Film noir lighting. High contrast black and white aesthetic, strong shadows, venetian blind patterns, mysterious and moody atmosphere.",
  // Stylized
  anime_modern: "Modern anime style with clean lines, expressive characters, dynamic poses, and vibrant color palettes. Contemporary Japanese animation aesthetic.",
  anime_ghibli: "Studio Ghibli inspired style. Soft, painterly backgrounds, expressive but grounded characters, attention to natural details, nostalgic and dreamlike quality.",
  pixel_art: "Pixel art style with deliberate limited resolution. Clear readable sprites, intentional color palette constraints, retro game aesthetic.",
  // Quality modifiers
  high_detail: "Maximum detail and resolution. Every element rendered with precision, textures clearly visible, sharp focus throughout, suitable for large format printing.",
  minimalist: "Minimalist composition. Essential elements only, generous negative space, clean and uncluttered, maximum impact through simplicity.",
  vintage_film: "Vintage film photography aesthetic. Film grain, slightly muted colors, light leaks optional, nostalgic analog camera feel.",
};

app.registerExtension({
  name: "Comfy.ZImageTextEncoder",

  async nodeCreated(node) {
    if (node.comfyClass !== "ZImageTextEncoder") return;

    const templateWidget = node.widgets?.find((w) => w.name === "template_preset");
    const systemPromptWidget = node.widgets?.find((w) => w.name === "system_prompt");

    if (!templateWidget || !systemPromptWidget) {
      console.warn("ZImageTextEncoder: Widgets not found", {
        template: !!templateWidget,
        system: !!systemPromptWidget,
        allWidgets: node.widgets?.map(w => w.name)
      });
      return;
    }

    console.log("ZImageTextEncoder: Setting up auto-fill for", node.id);

    // Store original callback
    const originalCallback = templateWidget.callback;

    // New callback that auto-fills system_prompt
    templateWidget.callback = function(value, ...args) {
      console.log("ZImageTextEncoder: Template changed to:", value);

      // Call original if exists
      if (originalCallback) {
        originalCallback.call(this, value, ...args);
      }

      // Auto-fill system_prompt if we have content for this template
      if (value in Z_IMAGE_TEMPLATES) {
        const content = Z_IMAGE_TEMPLATES[value];
        console.log("ZImageTextEncoder: Auto-filling system_prompt");
        systemPromptWidget.value = content;

        // Force widget update
        if (systemPromptWidget.inputEl) {
          systemPromptWidget.inputEl.value = content;
        }

        node.setDirtyCanvas(true, true);
      } else {
        // Template exists in Python (loaded from file) but not in JS auto-fill list
        // This is fine - user can edit system_prompt manually
        console.log("ZImageTextEncoder: Template", value, "loaded from file (edit system_prompt manually)");
      }
    };

    // Also intercept direct value changes via setter
    let currentValue = templateWidget.value;
    Object.defineProperty(templateWidget, 'value', {
      get() {
        return currentValue;
      },
      set(newValue) {
        const oldValue = currentValue;
        currentValue = newValue;
        if (oldValue !== newValue && templateWidget.callback) {
          templateWidget.callback(newValue);
        }
      },
      configurable: true
    });
  },
});
