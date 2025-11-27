/**
 * Z-Image Text Encoder UI Extension
 * Auto-fills system_prompt when template_preset changes
 */

import { app } from "../../../scripts/app.js";

// Z-Image templates - must match nodes/templates/z_image_*.md files
// Template name = filename without 'z_image_' prefix and '.md' suffix
const Z_IMAGE_TEMPLATES = {
  none: "",
  default: "",
  photorealistic: "Generate a photorealistic image with accurate lighting, natural textures, and realistic details. Pay attention to proper shadows, reflections, and material properties.",
  artistic: "Generate an artistic, aesthetically pleasing image with creative composition, harmonious colors, and visual balance. Focus on artistic expression and visual impact.",
  bilingual_text: "Generate an image with accurate text rendering. Support both English and Chinese text with clear, readable typography.",
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

      // Auto-fill system_prompt
      if (value in Z_IMAGE_TEMPLATES) {
        const content = Z_IMAGE_TEMPLATES[value];
        console.log("ZImageTextEncoder: Setting system_prompt to:", content.substring(0, 50) + "...");
        systemPromptWidget.value = content;

        // Force widget update
        if (systemPromptWidget.inputEl) {
          systemPromptWidget.inputEl.value = content;
        }

        node.setDirtyCanvas(true, true);
      } else {
        console.log("ZImageTextEncoder: Unknown template:", value);
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
