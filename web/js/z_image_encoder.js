/**
 * Z-Image Text Encoder UI Extension
 * Updates system_prompt value when template_preset changes
 *
 * Templates loaded from Python API - single source of truth
 * Endpoint: /api/z_image_templates (registered in __init__.py)
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Template cache - fetched from API on first use
let Z_IMAGE_TEMPLATES = null;
let templatesFetchPromise = null;

async function fetchTemplates() {
  if (Z_IMAGE_TEMPLATES !== null) {
    return Z_IMAGE_TEMPLATES;
  }

  if (templatesFetchPromise) {
    return templatesFetchPromise;
  }

  templatesFetchPromise = api
    .fetchApi("/api/z_image_templates")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      Z_IMAGE_TEMPLATES = data;
      Z_IMAGE_TEMPLATES.none = ""; // Ensure 'none' maps to empty
      return Z_IMAGE_TEMPLATES;
    })
    .catch((err) => {
      console.warn("ZImageTextEncoder: API fetch failed, using fallback:", err);
      Z_IMAGE_TEMPLATES = { none: "" }; // Python fallback will handle templates
      return Z_IMAGE_TEMPLATES;
    });

  return templatesFetchPromise;
}

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
      setTimeout(async () => {
        const templateWidget = node.widgets?.find(
          (w) => w.name === "template_preset"
        );
        const systemPromptWidget = node.widgets?.find(
          (w) => w.name === "system_prompt"
        );

        if (!templateWidget || !systemPromptWidget) {
          return;
        }

        // Fetch templates from API
        const templates = await fetchTemplates();

        // Store original callback
        const originalCallback = templateWidget.callback;

        // Override callback to update system_prompt field
        templateWidget.callback = function (value) {
          // Call original callback if exists
          if (originalCallback) {
            originalCallback.call(this, value);
          }

          // Update system_prompt based on preset
          if (templates[value] !== undefined) {
            systemPromptWidget.value = templates[value] || "";
          }

          // Mark node as needing update
          node.setDirtyCanvas(true, true);
        };

        // Trigger initial update if there's a value
        if (
          templateWidget.value &&
          templates[templateWidget.value] !== undefined
        ) {
          systemPromptWidget.value = templates[templateWidget.value] || "";
        }
      }, 10);

      return ret;
    };
  },
});
