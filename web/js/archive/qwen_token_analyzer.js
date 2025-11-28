/**
 * Qwen Token Analyzer UI Extension
 * Provides visual tokenization analysis and spatial coordinate tools
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

// Token definitions from analysis
const QWEN_TOKENS = {
    VISION: {
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
        "<|image_pad|>": 151655,
        "<|video_pad|>": 151656,
        "<|vision_pad|>": 151654
    },
    SPATIAL: {
        "<|object_ref_start|>": 151646,
        "<|object_ref_end|>": 151647,
        "<|box_start|>": 151648,
        "<|box_end|>": 151649,
        "<|quad_start|>": 151650,
        "<|quad_end|>": 151651
    },
    CHAT: {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645
    },
    CONTROL: {
        "<|endoftext|>": 151643
    }
};

// Flatten all tokens for easy lookup
const ALL_TOKENS = Object.assign({}, ...Object.values(QWEN_TOKENS));

class TokenAnalyzer {
    constructor() {
        this.tokenCounts = {};
        this.sequences = [];
    }

    analyzeText(text) {
        const analysis = {
            totalTokens: 0,
            specialTokens: 0,
            visionTokens: 0,
            spatialTokens: 0,
            sequences: [],
            errors: []
        };

        // Simple token counting (approximation without actual tokenizer)
        const words = text.split(/\s+/).filter(w => w.length > 0);
        analysis.totalTokens = words.length;

        // Count special tokens
        for (const [token, id] of Object.entries(ALL_TOKENS)) {
            const count = (text.match(new RegExp(this.escapeRegex(token), 'g')) || []).length;
            if (count > 0) {
                analysis.specialTokens += count;
                
                if (Object.values(QWEN_TOKENS.VISION).includes(id)) {
                    analysis.visionTokens += count;
                } else if (Object.values(QWEN_TOKENS.SPATIAL).includes(id)) {
                    analysis.spatialTokens += count;
                }
            }
        }

        // Analyze sequences
        analysis.sequences = this.findSequences(text);
        analysis.errors = this.validateSequences(analysis.sequences);

        return analysis;
    }

    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    findSequences(text) {
        const sequences = [];
        
        // Vision sequences
        const visionRegex = /<\|vision_start\|>(.*?)<\|vision_end\|>/g;
        let match;
        while ((match = visionRegex.exec(text)) !== null) {
            sequences.push({
                type: 'vision',
                content: match[0],
                inner: match[1],
                start: match.index,
                end: match.index + match[0].length
            });
        }

        // Spatial references
        const boxRegex = /<\|box_start\|>(.*?)<\|box_end\|>/g;
        while ((match = boxRegex.exec(text)) !== null) {
            sequences.push({
                type: 'box',
                content: match[0],
                coordinates: match[1],
                start: match.index,
                end: match.index + match[0].length
            });
        }

        // Object references
        const objRegex = /<\|object_ref_start\|>(.*?)<\|object_ref_end\|>/g;
        while ((match = objRegex.exec(text)) !== null) {
            sequences.push({
                type: 'object_ref',
                content: match[0],
                object: match[1],
                start: match.index,
                end: match.index + match[0].length
            });
        }

        return sequences.sort((a, b) => a.start - b.start);
    }

    validateSequences(sequences) {
        const errors = [];
        
        sequences.forEach((seq, index) => {
            if (seq.type === 'box') {
                const coords = seq.coordinates.split(',').map(s => s.trim());
                if (coords.length !== 4) {
                    errors.push(`Box sequence ${index}: Expected 4 coordinates, got ${coords.length}`);
                } else {
                    coords.forEach((coord, i) => {
                        const num = parseFloat(coord);
                        if (isNaN(num)) {
                            errors.push(`Box sequence ${index}: Invalid coordinate at position ${i}: "${coord}"`);
                        }
                    });
                }
            }
        });

        return errors;
    }

    generateTemplatePrompt(type, customText = "") {
        const templates = {
            basic_vision: `${customText} <|vision_start|><|image_pad|><|vision_end|>`,
            chat_vision: `<|im_start|>user\n${customText} <|vision_start|><|image_pad|><|vision_end|><|im_end|>`,
            spatial_edit: `Edit the <|object_ref_start|>object<|object_ref_end|> at <|box_start|>100,100,200,200<|box_end|>: ${customText}`,
            full_template: `<|im_start|>user\nEdit the <|object_ref_start|>target<|object_ref_end|> at <|box_start|>0,0,100,100<|box_end|> in this image: <|vision_start|><|image_pad|><|vision_end|>\n${customText}<|im_end|>`
        };
        
        return templates[type] || customText;
    }
}


// Create widget for token analysis display
function createTokenAnalysisWidget(node, inputName, inputData, app) {
    const widget = {
        name: inputName,
        type: "token_analysis",
        value: "",
        draw: function(ctx, node, widgetWidth, y, widgetHeight) {
            const analysis = new TokenAnalyzer().analyzeText(this.value || "");
            
            ctx.fillStyle = "#333";
            ctx.font = "12px Arial";
            
            let lineY = y + 15;
            const lineHeight = 16;
            
            ctx.fillText(`Tokens: ${analysis.totalTokens} | Special: ${analysis.specialTokens}`, 6, lineY);
            lineY += lineHeight;
            
            ctx.fillText(`Vision: ${analysis.visionTokens} | Spatial: ${analysis.spatialTokens}`, 6, lineY);
            lineY += lineHeight;
            
            if (analysis.errors.length > 0) {
                ctx.fillStyle = "#ff4444";
                ctx.fillText(`Errors: ${analysis.errors.length}`, 6, lineY);
                lineY += lineHeight;
            }
            
            return widgetHeight;
        },
        computeSize: function(width) {
            return [width, 60];
        }
    };
    
    node.addWidget("text", inputName, "", function(v) {
        widget.value = v;
        node.setDirtyCanvas(true, false);
    });
    
    return widget;
}

// Register the extension
app.registerExtension({
    name: "Comfy.QwenTokenAnalyzer",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add token analysis to QwenVLTextEncoder nodes
        if (nodeData.name === "QwenVLTextEncoder") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const node = this;
                
                // Add token analysis button
                setTimeout(() => {
                    const analyzeBtn = node.addWidget("button", "Analyze Tokens", "analyze", () => {
                        showTokenAnalysisDialog(node);
                    });
                }, 10);
                
                return ret;
            };
        }
    }
});

function showTokenAnalysisDialog(node) {
    const dialog = $el("div", {
        parent: document.body,
        style: {
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            background: "rgba(20, 20, 20, 0.95)",
            border: "1px solid rgba(255, 255, 255, 0.2)",
            padding: "20px",
            zIndex: 10000,
            maxWidth: "600px",
            maxHeight: "80vh",
            overflow: "auto",
            borderRadius: "8px",
            backdropFilter: "blur(10px)",
            boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
            fontFamily: "system-ui, -apple-system, sans-serif",
            color: "rgba(255, 255, 255, 0.9)"
        }
    });

    const analyzer = new TokenAnalyzer();
    
    // Get current text from node's text widget
    const textWidget = node.widgets?.find(w => w.name === "text" || w.name === "prompt");
    const currentText = textWidget?.value || "";
    
    const analysis = analyzer.analyzeText(currentText);
    
    dialog.innerHTML = `
        <h3 style="margin: 0 0 15px 0; color: rgba(255, 255, 255, 0.9); font-size: 18px; font-weight: 500;">Token Analysis</h3>
        
        <div style="
            background: rgba(255, 255, 255, 0.05);
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        ">
            <strong style="color: rgba(255, 255, 255, 0.9);">Statistics:</strong><br>
            <span style="font-family: 'Monaco', 'Menlo', monospace; font-size: 13px;">
                Total Tokens (approx): ${analysis.totalTokens}<br>
                Special Tokens: ${analysis.specialTokens}<br>
                Vision Tokens: ${analysis.visionTokens}<br>
                Spatial Tokens: ${analysis.spatialTokens}
            </span>
        </div>
        
        ${analysis.sequences.length > 0 ? `
            <div style="margin-bottom: 15px;">
                <strong style="color: rgba(255, 255, 255, 0.9);">Sequences Found:</strong>
                <div style="
                    background: rgba(255, 255, 255, 0.05);
                    padding: 8px;
                    border-radius: 4px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    font-family: 'Monaco', 'Menlo', monospace;
                    font-size: 12px;
                    max-height: 200px;
                    overflow-y: auto;
                    margin-top: 8px;
                ">
                    ${analysis.sequences.map(seq => 
                        `<div style="margin-bottom: 4px; color: rgba(255, 255, 255, 0.7);">
                            ${seq.type}: <code style="background: rgba(255, 255, 255, 0.1); padding: 1px 4px; border-radius: 2px;">${seq.content}</code>
                        </div>`
                    ).join('')}
                </div>
            </div>
        ` : ''}
        
        ${analysis.errors.length > 0 ? `
            <div style="margin-bottom: 15px;">
                <strong style="color: #ff6666;">Errors:</strong>
                <div style="
                    background: rgba(255, 0, 0, 0.1);
                    padding: 8px;
                    border-radius: 4px;
                    border: 1px solid rgba(255, 0, 0, 0.2);
                    margin-top: 8px;
                ">
                    ${analysis.errors.map(err => 
                        `<div style="color: #ff6666; font-size: 13px; margin-bottom: 4px;">• ${err}</div>`
                    ).join('')}
                </div>
            </div>
        ` : ''}
        
        <div style="margin-bottom: 15px;">
            <strong style="color: rgba(255, 255, 255, 0.9);">Template Generator:</strong><br>
            <select id="templateType" style="
                width: 100%;
                margin: 8px 0;
                padding: 6px;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                color: white;
                font-size: 12px;
            ">
                <option value="basic_vision">Basic Vision</option>
                <option value="chat_vision">Chat + Vision</option>
                <option value="spatial_edit">Spatial Edit</option>
                <option value="full_template">Full Template</option>
            </select><br>
            <input type="text" id="customText" placeholder="Custom text..." style="
                width: 100%;
                padding: 6px;
                margin: 5px 0;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                color: white;
                font-size: 12px;
            "><br>
            <button id="generateBtn" style="
                padding: 8px 16px;
                margin: 5px 0;
                background: rgba(0, 120, 255, 0.8);
                color: white;
                border: 1px solid rgba(0, 120, 255, 0.5);
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            ">Generate Template</button>
        </div>
        
        <textarea id="templateOutput" style="
            width: 100%;
            height: 100px;
            margin-bottom: 15px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 11px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            color: white;
            resize: vertical;
        " readonly></textarea>
        
        <div style="display: flex; gap: 6px; justify-content: flex-end;">
            <button id="useTemplate" style="
                padding: 6px 12px;
                background: rgba(76, 175, 80, 0.8);
                color: white;
                border: 1px solid rgba(76, 175, 80, 0.5);
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            ">Use Template</button>
            <button id="closeBtn" style="
                padding: 6px 12px;
                background: rgba(244, 67, 54, 0.8);
                color: white;
                border: 1px solid rgba(244, 67, 54, 0.5);
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            ">Close</button>
        </div>
    `;
    
    const templateOutput = dialog.querySelector('#templateOutput');
    const customText = dialog.querySelector('#customText');
    const templateType = dialog.querySelector('#templateType');
    
    dialog.querySelector('#generateBtn').onclick = () => {
        const template = analyzer.generateTemplatePrompt(templateType.value, customText.value);
        templateOutput.value = template;
    };
    
    dialog.querySelector('#useTemplate').onclick = () => {
        if (textWidget && templateOutput.value) {
            textWidget.value = templateOutput.value;
            node.setDirtyCanvas(true, true);
        }
        document.body.removeChild(dialog);
    };
    
    dialog.querySelector('#closeBtn').onclick = () => {
        document.body.removeChild(dialog);
    };
}

