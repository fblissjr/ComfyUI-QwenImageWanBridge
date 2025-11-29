/**
 * Unified Template Auto-fill Extension
 * Updates system prompt and thinking fields when template preset changes
 *
 * Supports: ZImageTextEncoder, ZImageTextEncoderSimple, HunyuanVideoTextEncoder
 * Templates loaded from Python API - single source of truth
 *
 * Z-Image templates can include:
 * - system_prompt (body text)
 * - add_think_block (boolean)
 * - thinking_content (string)
 * - assistant_content (string)
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Configuration for supported encoder nodes
const ENCODER_CONFIGS = {
  ZImageTextEncoder: {
    apiEndpoint: "/api/z_image_templates",
    templateWidget: "template_preset",
    systemWidget: "system_prompt",
    // Z-Image extended fields
    thinkBlockWidget: "add_think_block",
    thinkContentWidget: "thinking_content",
    assistantWidget: "assistant_content",
  },
  ZImageTextEncoderSimple: {
    apiEndpoint: "/api/z_image_templates",
    templateWidget: "template_preset",
    systemWidget: "system_prompt",
    // Z-Image extended fields
    thinkBlockWidget: "add_think_block",
    thinkContentWidget: "thinking_content",
    assistantWidget: "assistant_content",
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

        // Override callback to update all template-driven fields
        templateWidget.callback = function (value) {
          // Call original callback if exists
          if (originalCallback) {
            originalCallback.call(this, value);
          }

          // Get template data
          const template = templates[value];
          if (template === undefined) return;

          // Handle both new format (object) and legacy format (string)
          const isObject = typeof template === "object" && template !== null;

          // Update system prompt (always)
          const systemValue = isObject
            ? template.system_prompt || ""
            : template || "";
          systemWidget.value = systemValue;

          // Update Z-Image extended fields (if configured and template has them)
          if (isObject && config.thinkBlockWidget) {
            const thinkBlockWidget = node.widgets?.find(
              (w) => w.name === config.thinkBlockWidget
            );
            const thinkContentWidget = node.widgets?.find(
              (w) => w.name === config.thinkContentWidget
            );
            const assistantWidget = node.widgets?.find(
              (w) => w.name === config.assistantWidget
            );

            // Fill add_think_block if template specifies it
            if (thinkBlockWidget && template.add_think_block !== undefined) {
              thinkBlockWidget.value = template.add_think_block;
            }

            // Fill thinking_content if template has it
            if (thinkContentWidget && template.thinking_content) {
              thinkContentWidget.value = template.thinking_content;
            }

            // Fill assistant_content if template has it
            if (assistantWidget && template.assistant_content) {
              assistantWidget.value = template.assistant_content;
            }
          }

          // Mark node as needing update
          node.setDirtyCanvas(true, true);
        };

        // Trigger initial update only if system_prompt is empty
        // (preserves user customizations from saved workflows)
        if (
          templateWidget.value &&
          templates[templateWidget.value] !== undefined &&
          !systemWidget.value // Only fill if currently empty
        ) {
          templateWidget.callback(templateWidget.value);
        }
      }, 10);

      return ret;
    };
  },
});
