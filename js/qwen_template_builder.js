/**
 * Qwen Template Builder UI Extension
 * Provides an intuitive interface for building custom Qwen prompts with templates
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Template presets with descriptions
const TEMPLATE_PRESETS = {
    "default_t2i": {
        name: "Default Text-to-Image",
        description: "Standard Qwen T2I template",
        template: `<|im_start|>system
Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
`,
        hasVisionTokens: false
    },
    "default_edit": {
        name: "Default Image Edit",
        description: "Standard Qwen image editing template",
        template: `<|im_start|>system
Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>
<|im_start|>assistant
`,
        hasVisionTokens: true
    },
    "artistic": {
        name: "Artistic Freedom",
        description: "Emphasizes creative interpretation",
        template: `<|im_start|>system
You are an experimental artist. Break conventions. Be bold and creative. Interpret the prompt with artistic freedom.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
`,
        hasVisionTokens: false
    },
    "photorealistic": {
        name: "Photorealistic",
        description: "Emphasizes realistic rendering",
        template: `<|im_start|>system
You are a camera. Capture reality with perfect accuracy. No artistic interpretation. Focus on photorealistic details, proper lighting, and accurate proportions.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
`,
        hasVisionTokens: false
    },
    "minimal_edit": {
        name: "Minimal Edit",
        description: "Preserve original, minimal changes",
        template: `<|im_start|>system
Make only the specific changes requested. Preserve all other aspects of the original image exactly.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>
<|im_start|>assistant
`,
        hasVisionTokens: true
    },
    "style_transfer": {
        name: "Style Transfer",
        description: "Apply artistic style to image",
        template: `<|im_start|>system
Transform the image into the specified artistic style while preserving the original composition and subjects.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>
<|im_start|>assistant
`,
        hasVisionTokens: true
    },
    "technical": {
        name: "Technical/Diagram",
        description: "Technical drawings and diagrams",
        template: `<|im_start|>system
Generate technical diagrams and schematics. Use clean lines, proper labels, annotations, and professional technical drawing standards.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
`,
        hasVisionTokens: false
    },
    "custom": {
        name: "Custom Template",
        description: "Build your own template",
        template: `<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{vision_tokens}{prompt}<|im_end|>
<|im_start|>assistant
`,
        hasVisionTokens: false,
        customizable: true
    },
    "raw": {
        name: "Raw Prompt (No Template)",
        description: "Direct prompt without any template",
        template: `{prompt}`,
        hasVisionTokens: false
    }
};

// Special tokens reference
const SPECIAL_TOKENS = {
    "<|im_start|>": "Start of message",
    "<|im_end|>": "End of message",
    "<|vision_start|>": "Start of vision input (image edit only)",
    "<|image_pad|>": "Image data placeholder",
    "<|vision_end|>": "End of vision input",
    "{prompt}": "User's prompt text",
    "{system_prompt}": "Custom system instructions",
    "{vision_tokens}": "Vision token block for custom templates"
};

app.registerExtension({
    name: "QwenTemplateBuilder",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "QwenTemplateBuilder") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const ret = onNodeCreated ? onNodeCreated.apply(this) : undefined;
            
            // Get widgets
            const templateWidget = this.widgets.find(w => w.name === "template_preset");
            const systemPromptWidget = this.widgets.find(w => w.name === "system_prompt");
            const includeVisionWidget = this.widgets.find(w => w.name === "include_vision_tokens");
            const previewWidget = this.widgets.find(w => w.name === "template_preview");
            const tokenRefWidget = this.widgets.find(w => w.name === "token_reference");
            
            // Add token reference display
            if (tokenRefWidget) {
                let refText = "SPECIAL TOKENS:\n";
                for (const [token, desc] of Object.entries(SPECIAL_TOKENS)) {
                    refText += `${token.padEnd(20)} - ${desc}\n`;
                }
                tokenRefWidget.value = refText;
                tokenRefWidget.disabled = true;
            }
            
            // Update preview when template changes
            const updatePreview = () => {
                if (!templateWidget || !previewWidget) return;
                
                const selectedTemplate = TEMPLATE_PRESETS[templateWidget.value];
                if (!selectedTemplate) return;
                
                let preview = selectedTemplate.template;
                
                // Handle custom template
                if (templateWidget.value === "custom" && systemPromptWidget) {
                    preview = preview.replace("{system_prompt}", systemPromptWidget.value || "Your custom system prompt here");
                    
                    if (includeVisionWidget && includeVisionWidget.value) {
                        preview = preview.replace("{vision_tokens}", "<|vision_start|><|image_pad|><|vision_end|>");
                    } else {
                        preview = preview.replace("{vision_tokens}", "");
                    }
                }
                
                // Show example with placeholder prompt
                preview = preview.replace("{prompt}", "Your prompt text here");
                
                previewWidget.value = preview;
            };
            
            // Set up change handlers
            if (templateWidget) {
                templateWidget.callback = () => {
                    const selectedTemplate = TEMPLATE_PRESETS[templateWidget.value];
                    
                    // Show/hide custom options
                    if (systemPromptWidget) {
                        systemPromptWidget.widget.hidden = templateWidget.value !== "custom";
                    }
                    if (includeVisionWidget) {
                        includeVisionWidget.widget.hidden = templateWidget.value !== "custom";
                    }
                    
                    updatePreview();
                };
            }
            
            if (systemPromptWidget) {
                systemPromptWidget.callback = updatePreview;
            }
            
            if (includeVisionWidget) {
                includeVisionWidget.callback = updatePreview;
            }
            
            // Initial preview update
            updatePreview();
            
            return ret;
        };
    },
    
    async getCustomWidgets(app) {
        return {
            TEMPLATE_PREVIEW(node, inputName, inputData, app) {
                const widget = ComfyWidgets.STRING(node, inputName, ["STRING", { multiline: true }], app);
                widget.widget.disabled = true;
                return widget;
            }
        };
    }
});