/**
 * Qwen Template Builder UI Extension
 * Updates custom_system value when template preset changes
 * Templates loaded from nodes/templates/*.md files via Python
 */

import { app } from "../../../scripts/app.js";

// Template system prompts - loaded from nodes/templates/*.md files
// This is kept in sync by Python's template loader
const TEMPLATE_SYSTEMS = {
  default_t2i:
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:",
  default_edit:
    "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
  multi_image_edit:
    "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
  artistic:
    "You are an experimental artist. Break conventions. Be bold and creative. Interpret the prompt with artistic freedom.",
  photorealistic:
    "You are a camera. Capture reality with perfect accuracy. No artistic interpretation. Focus on photorealistic details, proper lighting, and accurate proportions.",
  minimal_edit:
    "Make only the specific changes requested. Preserve all other aspects of the original image exactly.",
  technical:
    "Generate technical diagrams and schematics. Use clean lines, proper labels, annotations, and professional technical drawing standards.",
  inpainting:
    "Replace or modify only the masked region according to the user's instruction. Preserve all other parts of the image exactly as they are. Blend the changes naturally with the surrounding context.",
  raw: "", // No template
};

app.registerExtension({
  name: "Comfy.QwenTemplateBuilder",

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
          (w) => w.name === "template_mode",
        );
        const customSystemWidget = node.widgets?.find(
          (w) => w.name === "custom_system",
        );

        if (!templateWidget || !customSystemWidget) {
          console.log("QwenTemplateBuilder: Missing widget", {
            templateWidget: !!templateWidget,
            customSystemWidget: !!customSystemWidget,
          });
          return;
        }

        console.log("QwenTemplateBuilder: Widgets found, setting up callback");

        // Store original callback
        const originalCallback = templateWidget.callback;

        // Override callback to update custom_system field
        templateWidget.callback = function (value) {
          console.log("QwenTemplateBuilder: Callback triggered", {
            value,
            hasTemplate: TEMPLATE_SYSTEMS[value] !== undefined,
          });

          // Call original callback if exists
          if (originalCallback) {
            originalCallback.call(this, value);
          }

          // Update custom_system based on preset
          if (TEMPLATE_SYSTEMS[value] !== undefined) {
            // raw template is empty string, others have content
            const newValue = TEMPLATE_SYSTEMS[value] || "";
            console.log(
              "QwenTemplateBuilder: Setting custom_system",
              newValue.substring(0, 50) + "...",
            );
            customSystemWidget.value = newValue;
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
