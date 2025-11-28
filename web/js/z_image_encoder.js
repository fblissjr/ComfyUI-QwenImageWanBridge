/**
 * Z-Image Text Encoder UI Extension
 * Updates system_prompt value when template_preset changes
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

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "ZImageTextEncoder") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const ret = onNodeCreated
        ? onNodeCreated.apply(this, arguments)
        : undefined;

      const node = this;

      // Find widgets after a short delay to ensure they're initialized
      setTimeout(() => {
        const templateWidget = node.widgets?.find(
          (w) => w.name === "template_preset"
        );
        const systemPromptWidget = node.widgets?.find(
          (w) => w.name === "system_prompt"
        );

        if (!templateWidget || !systemPromptWidget) {
          console.log("ZImageTextEncoder: Missing widget", {
            templateWidget: !!templateWidget,
            systemPromptWidget: !!systemPromptWidget,
          });
          return;
        }

        console.log("ZImageTextEncoder: Widgets found, setting up callback");

        // Store original callback
        const originalCallback = templateWidget.callback;

        // Override callback to update system_prompt field
        templateWidget.callback = function (value) {
          console.log("ZImageTextEncoder: Callback triggered", {
            value,
            hasTemplate: Z_IMAGE_TEMPLATES[value] !== undefined,
          });

          // Call original callback if exists
          if (originalCallback) {
            originalCallback.call(this, value);
          }

          // Update system_prompt based on preset
          if (Z_IMAGE_TEMPLATES[value] !== undefined) {
            const newValue = Z_IMAGE_TEMPLATES[value] || "";
            console.log(
              "ZImageTextEncoder: Setting system_prompt to:",
              newValue.substring(0, 50) + (newValue.length > 50 ? "..." : "")
            );
            systemPromptWidget.value = newValue;
          } else {
            // Template loaded from file but not in JS - user can edit manually
            console.log(
              "ZImageTextEncoder: Template",
              value,
              "loaded from file (edit system_prompt manually)"
            );
          }

          // Mark node as needing update
          node.setDirtyCanvas(true, true);
        };

        // Trigger initial update if there's a value
        if (templateWidget.value) {
          templateWidget.callback(templateWidget.value);
        }
      }, 10);

      return ret;
    };
  },
});
