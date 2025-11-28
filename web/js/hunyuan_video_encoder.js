/**
 * HunyuanVideo Text Encoder UI Extension
 * Updates custom_system_prompt value when template_preset changes
 *
 * Templates loaded from Python API - single source of truth
 * Endpoint: /api/hunyuan_video_templates (registered in __init__.py)
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Template cache - fetched from API on first use
let HUNYUAN_VIDEO_TEMPLATES = null;
let templatesFetchPromise = null;

async function fetchTemplates() {
  if (HUNYUAN_VIDEO_TEMPLATES !== null) {
    return HUNYUAN_VIDEO_TEMPLATES;
  }

  if (templatesFetchPromise) {
    return templatesFetchPromise;
  }

  templatesFetchPromise = api
    .fetchApi("/api/hunyuan_video_templates")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      HUNYUAN_VIDEO_TEMPLATES = data;
      HUNYUAN_VIDEO_TEMPLATES.none = ""; // Ensure 'none' maps to empty
      return HUNYUAN_VIDEO_TEMPLATES;
    })
    .catch((err) => {
      console.warn(
        "HunyuanVideoTextEncoder: API fetch failed, using fallback:",
        err
      );
      HUNYUAN_VIDEO_TEMPLATES = { none: "" }; // Python fallback will handle templates
      return HUNYUAN_VIDEO_TEMPLATES;
    });

  return templatesFetchPromise;
}

app.registerExtension({
  name: "Comfy.HunyuanVideoTextEncoder",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "HunyuanVideoTextEncoder") return;

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
          (w) => w.name === "custom_system_prompt"
        );

        if (!templateWidget || !systemPromptWidget) {
          return;
        }

        // Fetch templates from API
        const templates = await fetchTemplates();

        // Store original callback
        const originalCallback = templateWidget.callback;

        // Override callback to update custom_system_prompt field
        templateWidget.callback = function (value) {
          // Call original callback if exists
          if (originalCallback) {
            originalCallback.call(this, value);
          }

          // Update custom_system_prompt based on preset
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
