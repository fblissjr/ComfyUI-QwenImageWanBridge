/**
 * Z-Image Text Encoder UI Extension
 * Auto-fills system_prompt when template_preset changes
 * Templates loaded from nodes/templates/z_image_*.md
 */

import { app } from "../../../scripts/app.js";

// Z-Image template system prompts - mirrors nodes/templates/z_image_*.md
// When you add a new template file, add it here too
const Z_IMAGE_TEMPLATES = {
  none: "",
  default:
    "",
  photorealistic:
    "Generate a photorealistic image with accurate lighting, natural textures, and realistic details. Pay attention to proper shadows, reflections, and material properties.",
  artistic:
    "Generate an artistic, aesthetically pleasing image with creative composition, harmonious colors, and visual balance. Focus on artistic expression and visual impact.",
  bilingual_text:
    "Generate an image with accurate text rendering. Support both English and Chinese text with clear, readable typography.",
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

      // Find widgets after short delay
      setTimeout(() => {
        const templateWidget = node.widgets?.find(
          (w) => w.name === "template_preset"
        );
        const systemPromptWidget = node.widgets?.find(
          (w) => w.name === "system_prompt"
        );

        if (!templateWidget || !systemPromptWidget) {
          console.log("ZImageTextEncoder: Missing widgets", {
            templateWidget: !!templateWidget,
            systemPromptWidget: !!systemPromptWidget,
          });
          return;
        }

        console.log("ZImageTextEncoder: Setting up template auto-fill");

        const originalCallback = templateWidget.callback;

        templateWidget.callback = function (value) {
          if (originalCallback) {
            originalCallback.call(this, value);
          }

          // Auto-fill system_prompt from template
          if (Z_IMAGE_TEMPLATES[value] !== undefined) {
            const newValue = Z_IMAGE_TEMPLATES[value] || "";
            console.log(
              "ZImageTextEncoder: Auto-filling system_prompt:",
              newValue.substring(0, 40) + (newValue.length > 40 ? "..." : "")
            );
            systemPromptWidget.value = newValue;
          }

          node.setDirtyCanvas(true, true);
        };

        // Initial fill if template already selected
        if (templateWidget.value && templateWidget.value !== "none") {
          templateWidget.callback(templateWidget.value);
        }
      }, 10);

      return ret;
    };
  },
});
