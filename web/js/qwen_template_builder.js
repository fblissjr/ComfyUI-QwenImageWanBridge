/**
 * Qwen Template Builder UI Extension - Simple Version
 * Updates system_prompt value when template preset changes
 */

import { app } from "../../../scripts/app.js";

// Template system prompts
const TEMPLATE_SYSTEMS = {
  default_edit:
    "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
  default_t2i:
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:",
  custom: null, // User provides
  raw: null, // No template
};

const TEMPLATE_VISION = {
  default_edit: true,
  default_t2i: false,
  artistic: false,
  photorealistic: false,
  minimal_edit: true,
  style_transfer: true,
  technical: false,
  custom: true, // Changed to true since default is now edit mode
  raw: false,
};

app.registerExtension({
  name: "Comfy.QwenTemplateBuilderSimple",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "QwenTemplateBuilder") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const ret = onNodeCreated
        ? onNodeCreated.apply(this, arguments)
        : undefined;

      const node = this;

      // Find widgets after a short delay
      setTimeout(() => {
        const templateWidget = node.widgets?.find(
          (w) => w.name === "template_preset",
        );
        const systemPromptWidget = node.widgets?.find(
          (w) => w.name === "system_prompt",
        );
        const visionTokenWidget = node.widgets?.find(
          (w) => w.name === "include_vision_tokens",
        );

        if (!templateWidget) return;

        // Store original callback
        const originalCallback = templateWidget.callback;

        // Override callback to update other widgets
        templateWidget.callback = function (value) {
          // Call original callback if exists
          if (originalCallback) {
            originalCallback.call(this, value);
          }

          // Update system prompt based on preset
          if (systemPromptWidget && TEMPLATE_SYSTEMS[value] !== undefined) {
            if (TEMPLATE_SYSTEMS[value] !== null) {
              systemPromptWidget.value = TEMPLATE_SYSTEMS[value];
            }
          }

          // Update vision tokens based on preset
          if (visionTokenWidget && TEMPLATE_VISION[value] !== undefined) {
            visionTokenWidget.value = TEMPLATE_VISION[value];
          }

          // Mark node as needing update
          node.setDirtyCanvas(true, true);
        };

        // Trigger initial update
        if (templateWidget.value) {
          templateWidget.callback(templateWidget.value);
        }
      }, 10);

      return ret;
    };
  },
});
