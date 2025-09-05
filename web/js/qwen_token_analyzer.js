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

class SpatialCoordinateEditor {
    constructor() {
        this.regions = [];
        this.canvas = null;
        this.ctx = null;
        this.activeRegion = null;
        this.isDragging = false;
    }

    createCanvas(width = 512, height = 512) {
        this.canvas = $el("canvas", {
            width: width,
            height: height,
            style: {
                border: "1px solid #ccc",
                cursor: "crosshair",
                maxWidth: "100%"
            }
        });
        
        this.ctx = this.canvas.getContext('2d');
        this.setupEventListeners();
        return this.canvas;
    }

    setupEventListeners() {
        let startX, startY;
        
        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
            this.isDragging = true;
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            const rect = this.canvas.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;
            
            this.redraw();
            this.drawRegion(startX, startY, currentX, currentY, 'rgba(0, 255, 0, 0.3)');
        });

        this.canvas.addEventListener('mouseup', (e) => {
            if (!this.isDragging) return;
            
            const rect = this.canvas.getBoundingClientRect();
            const endX = e.clientX - rect.left;
            const endY = e.clientY - rect.top;
            
            const region = {
                x1: Math.min(startX, endX),
                y1: Math.min(startY, endY),
                x2: Math.max(startX, endX),
                y2: Math.max(startY, endY),
                label: `region_${this.regions.length + 1}`
            };
            
            this.regions.push(region);
            this.redraw();
            this.isDragging = false;
            
            // Trigger callback if set
            if (this.onRegionCreated) {
                this.onRegionCreated(region);
            }
        });
    }

    drawRegion(x1, y1, x2, y2, color = 'rgba(255, 0, 0, 0.3)') {
        this.ctx.fillStyle = color;
        this.ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        this.ctx.strokeStyle = 'red';
        this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    redraw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.regions.forEach(region => {
            this.drawRegion(region.x1, region.y1, region.x2, region.y2);
            
            // Draw label
            this.ctx.fillStyle = 'black';
            this.ctx.font = '12px Arial';
            this.ctx.fillText(region.label, region.x1 + 2, region.y1 + 14);
        });
    }

    generateToken(region) {
        return `<|object_ref_start|>${region.label}<|object_ref_end|> at <|box_start|>${region.x1},${region.y1},${region.x2},${region.y2}<|box_end|>`;
    }

    clearRegions() {
        this.regions = [];
        this.redraw();
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
                    
                    // Add spatial editor button
                    const spatialBtn = node.addWidget("button", "Spatial Editor", "spatial", () => {
                        showSpatialEditorDialog(node);
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
            background: "white",
            border: "2px solid #333",
            padding: "20px",
            zIndex: 10000,
            maxWidth: "600px",
            maxHeight: "80vh",
            overflow: "auto",
            borderRadius: "8px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.3)"
        }
    });

    const analyzer = new TokenAnalyzer();
    
    // Get current text from node's text widget
    const textWidget = node.widgets?.find(w => w.name === "text" || w.name === "prompt");
    const currentText = textWidget?.value || "";
    
    const analysis = analyzer.analyzeText(currentText);
    
    dialog.innerHTML = `
        <h3 style="margin: 0 0 15px 0; color: #333;">Token Analysis</h3>
        
        <div style="margin-bottom: 15px;">
            <strong>Statistics:</strong><br>
            Total Tokens (approx): ${analysis.totalTokens}<br>
            Special Tokens: ${analysis.specialTokens}<br>
            Vision Tokens: ${analysis.visionTokens}<br>
            Spatial Tokens: ${analysis.spatialTokens}
        </div>
        
        ${analysis.sequences.length > 0 ? `
            <div style="margin-bottom: 15px;">
                <strong>Sequences Found:</strong>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    ${analysis.sequences.map(seq => 
                        `<li>${seq.type}: <code>${seq.content}</code></li>`
                    ).join('')}
                </ul>
            </div>
        ` : ''}
        
        ${analysis.errors.length > 0 ? `
            <div style="margin-bottom: 15px; color: #d32f2f;">
                <strong>Errors:</strong>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    ${analysis.errors.map(err => `<li>${err}</li>`).join('')}
                </ul>
            </div>
        ` : ''}
        
        <div style="margin-bottom: 15px;">
            <strong>Template Generator:</strong><br>
            <select id="templateType" style="margin: 5px 0; padding: 5px;">
                <option value="basic_vision">Basic Vision</option>
                <option value="chat_vision">Chat + Vision</option>
                <option value="spatial_edit">Spatial Edit</option>
                <option value="full_template">Full Template</option>
            </select><br>
            <input type="text" id="customText" placeholder="Custom text..." style="width: 100%; padding: 5px; margin: 5px 0;"><br>
            <button id="generateBtn" style="padding: 8px 16px; margin: 5px 0;">Generate Template</button>
        </div>
        
        <textarea id="templateOutput" style="width: 100%; height: 100px; margin-bottom: 15px; font-family: monospace;" readonly></textarea>
        
        <div style="text-align: right;">
            <button id="useTemplate" style="padding: 8px 16px; margin: 5px; background: #4caf50; color: white; border: none; border-radius: 4px;">Use Template</button>
            <button id="closeBtn" style="padding: 8px 16px; margin: 5px; background: #f44336; color: white; border: none; border-radius: 4px;">Close</button>
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

function showSpatialEditorDialog(node) {
    const dialog = $el("div", {
        parent: document.body,
        style: {
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            background: "white",
            border: "2px solid #333",
            padding: "20px",
            zIndex: 10000,
            width: "700px",
            maxHeight: "80vh",
            overflow: "auto",
            borderRadius: "8px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.3)"
        }
    });

    const spatialEditor = new SpatialCoordinateEditor();
    const canvas = spatialEditor.createCanvas(512, 512);
    
    dialog.innerHTML = `
        <h3 style="margin: 0 0 15px 0; color: #333;">Spatial Coordinate Editor</h3>
        <p style="margin-bottom: 15px; color: #666;">Click and drag to create bounding boxes. Generated tokens will appear below.</p>
        <div style="text-align: center; margin-bottom: 15px;"></div>
        <div style="margin-bottom: 15px;">
            <button id="clearRegions" style="padding: 8px 16px; margin: 5px; background: #ff9800; color: white; border: none; border-radius: 4px;">Clear Regions</button>
            <button id="exportTokens" style="padding: 8px 16px; margin: 5px; background: #2196f3; color: white; border: none; border-radius: 4px;">Export Tokens</button>
        </div>
        <textarea id="tokensOutput" style="width: 100%; height: 120px; margin-bottom: 15px; font-family: monospace;" readonly placeholder="Generated spatial tokens will appear here..."></textarea>
        <div style="text-align: right;">
            <button id="useTokens" style="padding: 8px 16px; margin: 5px; background: #4caf50; color: white; border: none; border-radius: 4px;">Use Tokens</button>
            <button id="closeBtn" style="padding: 8px 16px; margin: 5px; background: #f44336; color: white; border: none; border-radius: 4px;">Close</button>
        </div>
    `;
    
    dialog.querySelector('div').appendChild(canvas);
    
    const tokensOutput = dialog.querySelector('#tokensOutput');
    
    spatialEditor.onRegionCreated = (region) => {
        const tokens = spatialEditor.regions.map(r => spatialEditor.generateToken(r)).join('\n');
        tokensOutput.value = tokens;
    };
    
    dialog.querySelector('#clearRegions').onclick = () => {
        spatialEditor.clearRegions();
        tokensOutput.value = '';
    };
    
    dialog.querySelector('#exportTokens').onclick = () => {
        const tokens = spatialEditor.regions.map(r => spatialEditor.generateToken(r)).join(' ');
        navigator.clipboard.writeText(tokens).then(() => {
            alert('Tokens copied to clipboard!');
        });
    };
    
    dialog.querySelector('#useTokens').onclick = () => {
        const textWidget = node.widgets?.find(w => w.name === "text" || w.name === "prompt");
        if (textWidget && tokensOutput.value) {
            const currentText = textWidget.value || "";
            textWidget.value = currentText + (currentText ? ' ' : '') + tokensOutput.value;
            node.setDirtyCanvas(true, true);
        }
        document.body.removeChild(dialog);
    };
    
    dialog.querySelector('#closeBtn').onclick = () => {
        document.body.removeChild(dialog);
    };
}