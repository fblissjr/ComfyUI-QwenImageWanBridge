/**
 * Unified Template Auto-fill Extension
 * Updates system prompt field when template preset changes
 *
 * Supports: ZImageTextEncoder, HunyuanVideoTextEncoder
 * Templates loaded from Python API - single source of truth
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Configuration for supported encoder nodes
const ENCODER_CONFIGS = {
  ZImageTextEncoder: {
    apiEndpoint: "/api/z_image_templates",
    templateWidget: "template_preset",
    systemWidget: "system_prompt",
  },
  HunyuanVideoTextEncoder: {
    apiEndpoint: "/api/hunyuan_video_templates",
    templateWidget: "template_preset",
    systemWidget: "custom_system_prompt",
  },
};

// Per-encoder template cache
const templateCache = {};
const fetchPromises = {};

async function fetchTemplates(encoderType) {
  const config = ENCODER_CONFIGS[encoderType];
  if (!config) return {};

  // Return cached templates if available
  if (templateCache[encoderType]) {
    return templateCache[encoderType];
  }

  // Return in-flight promise if fetch is already in progress
  if (fetchPromises[encoderType]) {
    return fetchPromises[encoderType];
  }

  // Fetch templates from API
  fetchPromises[encoderType] = api
    .fetchApi(config.apiEndpoint)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      data.none = ""; // Ensure 'none' maps to empty
      templateCache[encoderType] = data;
      return data;
    })
    .catch((err) => {
      console.warn(`${encoderType}: API fetch failed, using fallback:`, err);
      templateCache[encoderType] = { none: "" }; // Python fallback will handle templates
      return templateCache[encoderType];
    });

  return fetchPromises[encoderType];
}

app.registerExtension({
  name: "Comfy.TemplateAutofill",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    const config = ENCODER_CONFIGS[nodeData.name];
    if (!config) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const ret = onNodeCreated
        ? onNodeCreated.apply(this, arguments)
        : undefined;

      const node = this;
      const encoderType = nodeData.name;

      // Find widgets after a short delay to ensure they're initialized
      setTimeout(async () => {
        const templateWidget = node.widgets?.find(
          (w) => w.name === config.templateWidget
        );
        const systemWidget = node.widgets?.find(
          (w) => w.name === config.systemWidget
        );

        if (!templateWidget || !systemWidget) {
          return;
        }

        // Fetch templates from API
        const templates = await fetchTemplates(encoderType);

        // Store original callback
        const originalCallback = templateWidget.callback;

        // Override callback to update system prompt field
        templateWidget.callback = function (value) {
          // Call original callback if exists
          if (originalCallback) {
            originalCallback.call(this, value);
          }

          // Update system prompt based on preset
          if (templates[value] !== undefined) {
            systemWidget.value = templates[value] || "";
          }

          // Mark node as needing update
          node.setDirtyCanvas(true, true);
        };

        // Trigger initial update if there's a value
        if (
          templateWidget.value &&
          templates[templateWidget.value] !== undefined
        ) {
          systemWidget.value = templates[templateWidget.value] || "";
        }
      }, 10);

      return ret;
    };
  },
});
