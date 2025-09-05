/**
 * Qwen Token Sequence Visualizer
 * Advanced visualization and analysis of token sequences
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

// Token color scheme for visualization
const TOKEN_COLORS = {
    VISION: {
        background: "#e3f2fd",
        border: "#2196f3",
        text: "#1565c0"
    },
    SPATIAL: {
        background: "#f3e5f5",
        border: "#9c27b0",
        text: "#7b1fa2"
    },
    CHAT: {
        background: "#e8f5e8",
        border: "#4caf50",
        text: "#2e7d32"
    },
    CONTROL: {
        background: "#ffebee",
        border: "#f44336",
        text: "#c62828"
    },
    CODE: {
        background: "#fff3e0",
        border: "#ff9800",
        text: "#e65100"
    },
    TOOL: {
        background: "#fce4ec",
        border: "#e91e63",
        text: "#ad1457"
    },
    REGULAR: {
        background: "#f5f5f5",
        border: "#9e9e9e",
        text: "#424242"
    }
};

// Token definitions with categories
const QWEN_TOKENS_CATEGORIZED = {
    "<|vision_start|>": "VISION",
    "<|vision_end|>": "VISION", 
    "<|image_pad|>": "VISION",
    "<|video_pad|>": "VISION",
    "<|vision_pad|>": "VISION",
    "<|object_ref_start|>": "SPATIAL",
    "<|object_ref_end|>": "SPATIAL",
    "<|box_start|>": "SPATIAL",
    "<|box_end|>": "SPATIAL",
    "<|quad_start|>": "SPATIAL",
    "<|quad_end|>": "SPATIAL",
    "<|im_start|>": "CHAT",
    "<|im_end|>": "CHAT",
    "<|endoftext|>": "CONTROL",
    "<|fim_prefix|>": "CODE",
    "<|fim_middle|>": "CODE",
    "<|fim_suffix|>": "CODE",
    "<|fim_pad|>": "CODE",
    "<|repo_name|>": "CODE",
    "<|file_sep|>": "CODE",
    "<tool_call>": "TOOL",
    "</tool_call>": "TOOL"
};

class TokenSequenceVisualizer {
    constructor(container) {
        this.container = container;
        this.tokens = [];
        this.sequences = [];
        this.currentText = "";
        this.highlightedTokens = new Set();
        
        this.createInterface();
    }
    
    createInterface() {
        this.container.innerHTML = `
            <div class="token-visualizer" style="
                border: 1px solid #ccc;
                border-radius: 8px;
                background: white;
                max-height: 400px;
                overflow-y: auto;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 12px;
            ">
                <div class="visualizer-header" style="
                    padding: 8px 12px;
                    background: #f8f9fa;
                    border-bottom: 1px solid #e9ecef;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div class="token-stats">
                        <span id="token-count">0 tokens</span>
                        <span style="margin-left: 10px;" id="special-count">0 special</span>
                    </div>
                    <div class="controls">
                        <button id="highlight-sequences" style="
                            padding: 4px 8px;
                            font-size: 11px;
                            border: 1px solid #007bff;
                            background: white;
                            color: #007bff;
                            border-radius: 4px;
                            cursor: pointer;
                        ">Highlight Sequences</button>
                        <button id="export-analysis" style="
                            padding: 4px 8px;
                            font-size: 11px;
                            border: 1px solid #28a745;
                            background: white;
                            color: #28a745;
                            border-radius: 4px;
                            cursor: pointer;
                            margin-left: 5px;
                        ">Export</button>
                    </div>
                </div>
                <div id="token-display" style="
                    padding: 12px;
                    line-height: 1.8;
                    word-wrap: break-word;
                "></div>
                <div id="sequence-legend" style="
                    padding: 8px 12px;
                    background: #f8f9fa;
                    border-top: 1px solid #e9ecef;
                    display: none;
                "></div>
            </div>
        `;
        
        this.tokenDisplay = this.container.querySelector('#token-display');
        this.tokenCount = this.container.querySelector('#token-count');
        this.specialCount = this.container.querySelector('#special-count');
        this.sequenceLegend = this.container.querySelector('#sequence-legend');
        
        // Event listeners
        this.container.querySelector('#highlight-sequences').onclick = () => {
            this.toggleSequenceHighlighting();
        };
        
        this.container.querySelector('#export-analysis').onclick = () => {
            this.exportAnalysis();
        };
    }
    
    analyzeText(text) {
        this.currentText = text;
        this.tokens = this.tokenizeText(text);
        this.sequences = this.findSequences(text);
        this.renderTokens();
        this.updateStats();
    }
    
    tokenizeText(text) {
        const tokens = [];
        let currentIndex = 0;
        let tokenId = 0;
        
        // Split text preserving special tokens
        const specialTokens = Object.keys(QWEN_TOKENS_CATEGORIZED);
        const tokenRegex = new RegExp(`(${specialTokens.map(t => this.escapeRegex(t)).join('|')})`, 'g');
        
        let lastIndex = 0;
        let match;
        
        while ((match = tokenRegex.exec(text)) !== null) {
            // Add regular text before special token
            if (match.index > lastIndex) {
                const regularText = text.substring(lastIndex, match.index);
                const words = regularText.split(/(\s+)/).filter(s => s.length > 0);
                
                words.forEach(word => {
                    if (word.trim()) {
                        tokens.push({
                            id: tokenId++,
                            text: word,
                            type: "REGULAR",
                            start: currentIndex,
                            end: currentIndex + word.length,
                            isSpecial: false
                        });
                    }
                    currentIndex += word.length;
                });
            }
            
            // Add special token
            tokens.push({
                id: tokenId++,
                text: match[0],
                type: QWEN_TOKENS_CATEGORIZED[match[0]],
                start: match.index,
                end: match.index + match[0].length,
                isSpecial: true
            });
            
            currentIndex = match.index + match[0].length;
            lastIndex = match.index + match[0].length;
        }
        
        // Add remaining regular text
        if (lastIndex < text.length) {
            const remainingText = text.substring(lastIndex);
            const words = remainingText.split(/(\s+)/).filter(s => s.length > 0);
            
            words.forEach(word => {
                if (word.trim()) {
                    tokens.push({
                        id: tokenId++,
                        text: word,
                        type: "REGULAR", 
                        start: currentIndex,
                        end: currentIndex + word.length,
                        isSpecial: false
                    });
                }
                currentIndex += word.length;
            });
        }
        
        return tokens;
    }
    
    findSequences(text) {
        const sequences = [];
        let sequenceId = 0;
        
        // Vision sequences
        const visionRegex = /<\|vision_start\|>(.*?)<\|vision_end\|>/g;
        let match;
        while ((match = visionRegex.exec(text)) !== null) {
            sequences.push({
                id: sequenceId++,
                type: 'vision',
                name: 'Vision Processing',
                start: match.index,
                end: match.index + match[0].length,
                content: match[0],
                innerContent: match[1],
                color: '#2196f3'
            });
        }
        
        // Spatial sequences
        const boxRegex = /<\|box_start\|>(.*?)<\|box_end\|>/g;
        while ((match = boxRegex.exec(text)) !== null) {
            sequences.push({
                id: sequenceId++,
                type: 'box',
                name: 'Bounding Box',
                start: match.index,
                end: match.index + match[0].length,
                content: match[0],
                coordinates: match[1],
                color: '#9c27b0'
            });
        }
        
        // Object reference sequences
        const objRegex = /<\|object_ref_start\|>(.*?)<\|object_ref_end\|>/g;
        while ((match = objRegex.exec(text)) !== null) {
            sequences.push({
                id: sequenceId++,
                type: 'object_ref',
                name: 'Object Reference',
                start: match.index,
                end: match.index + match[0].length,
                content: match[0],
                objectName: match[1],
                color: '#9c27b0'
            });
        }
        
        // Chat sequences
        const chatRegex = /<\|im_start\|>(.*?)<\|im_end\|>/gs;
        while ((match = chatRegex.exec(text)) !== null) {
            sequences.push({
                id: sequenceId++,
                type: 'chat',
                name: 'Chat Message',
                start: match.index,
                end: match.index + match[0].length,
                content: match[0],
                messageContent: match[1],
                color: '#4caf50'
            });
        }
        
        return sequences.sort((a, b) => a.start - b.start);
    }
    
    renderTokens() {
        const tokenElements = this.tokens.map(token => {
            const colors = TOKEN_COLORS[token.type] || TOKEN_COLORS.REGULAR;
            const isHighlighted = this.highlightedTokens.has(token.id);
            
            return `<span 
                class="token" 
                data-token-id="${token.id}"
                data-token-type="${token.type}"
                style="
                    display: inline-block;
                    padding: 2px 4px;
                    margin: 1px;
                    background: ${isHighlighted ? '#ffeb3b' : colors.background};
                    border: 1px solid ${colors.border};
                    color: ${colors.text};
                    border-radius: 3px;
                    cursor: pointer;
                    font-weight: ${token.isSpecial ? 'bold' : 'normal'};
                    position: relative;
                "
                title="${token.isSpecial ? `Special token: ${token.text}` : token.text}"
                onmouseover="this.style.opacity='0.8'"
                onmouseout="this.style.opacity='1'"
                onclick="window.qwenTokenVisualizer?.toggleTokenHighlight(${token.id})"
            >${this.escapeHtml(token.text)}</span>`;
        }).join('');
        
        this.tokenDisplay.innerHTML = tokenElements;
        
        // Store reference for global access
        window.qwenTokenVisualizer = this;
    }
    
    toggleTokenHighlight(tokenId) {
        if (this.highlightedTokens.has(tokenId)) {
            this.highlightedTokens.delete(tokenId);
        } else {
            this.highlightedTokens.add(tokenId);
        }
        this.renderTokens();
    }
    
    toggleSequenceHighlighting() {
        const isShowing = this.sequenceLegend.style.display !== 'none';
        
        if (isShowing) {
            // Hide sequence highlighting
            this.sequenceLegend.style.display = 'none';
            this.renderTokens();
        } else {
            // Show sequence highlighting
            this.sequenceLegend.style.display = 'block';
            this.renderSequenceHighlighting();
        }
    }
    
    renderSequenceHighlighting() {
        // Create legend
        const legendItems = this.sequences.map(seq => 
            `<span style="
                display: inline-block;
                margin: 2px 8px 2px 0;
                padding: 2px 6px;
                background: ${seq.color}20;
                border: 1px solid ${seq.color};
                border-radius: 3px;
                font-size: 11px;
            ">${seq.name}</span>`
        ).join('');
        
        this.sequenceLegend.innerHTML = `<strong>Sequences:</strong> ${legendItems}`;
        
        // Highlight tokens that are part of sequences
        this.tokens.forEach(token => {
            const belongsToSequences = this.sequences.filter(seq => 
                token.start >= seq.start && token.end <= seq.end
            );
            
            if (belongsToSequences.length > 0) {
                const tokenElement = this.tokenDisplay.querySelector(`[data-token-id="${token.id}"]`);
                if (tokenElement) {
                    const seq = belongsToSequences[0]; // Use first sequence if overlapping
                    tokenElement.style.boxShadow = `0 0 0 2px ${seq.color}40`;
                    tokenElement.style.position = 'relative';
                }
            }
        });
    }
    
    updateStats() {
        const specialTokens = this.tokens.filter(t => t.isSpecial);
        this.tokenCount.textContent = `${this.tokens.length} tokens`;
        this.specialCount.textContent = `${specialTokens.length} special`;
    }
    
    exportAnalysis() {
        const analysis = {
            text: this.currentText,
            tokenCount: this.tokens.length,
            specialTokenCount: this.tokens.filter(t => t.isSpecial).length,
            tokens: this.tokens,
            sequences: this.sequences,
            tokensByType: this.getTokensByType(),
            efficiency: this.calculateEfficiency()
        };
        
        const blob = new Blob([JSON.stringify(analysis, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `qwen_token_analysis_${new Date().toISOString().slice(0, 19)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    getTokensByType() {
        const byType = {};
        this.tokens.forEach(token => {
            if (!byType[token.type]) {
                byType[token.type] = 0;
            }
            byType[token.type]++;
        });
        return byType;
    }
    
    calculateEfficiency() {
        const specialTokens = this.tokens.filter(t => t.isSpecial).length;
        const regularTokens = this.tokens.length - specialTokens;
        
        return {
            specialTokenRatio: specialTokens / this.tokens.length,
            averageTokenLength: this.tokens.reduce((sum, t) => sum + t.text.length, 0) / this.tokens.length,
            sequenceDensity: this.sequences.length / this.tokens.length,
            compressionRatio: this.currentText.length / this.tokens.length
        };
    }
    
    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Enhanced extension for QwenTokenDebugger node
app.registerExtension({
    name: "Comfy.QwenTokenVisualizer",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "QwenTokenDebugger") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const node = this;
                
                // Add token visualizer after a delay
                setTimeout(() => {
                    // Create visualizer container
                    const visualizerContainer = $el("div", {
                        style: {
                            marginTop: "10px",
                            border: "1px solid #ccc",
                            borderRadius: "4px",
                            minHeight: "200px"
                        }
                    });
                    
                    // Add to node
                    node.addDOMWidget("tokenVisualizer", "TOKEN_VISUALIZER", visualizerContainer);
                    
                    // Create visualizer
                    const visualizer = new TokenSequenceVisualizer(visualizerContainer);
                    
                    // Find text input widget and set up auto-analysis
                    const textWidget = node.widgets?.find(w => w.name === "text");
                    if (textWidget) {
                        const originalCallback = textWidget.callback;
                        textWidget.callback = function(value) {
                            if (originalCallback) {
                                originalCallback.call(this, value);
                            }
                            
                            // Update visualizer
                            visualizer.analyzeText(value || "");
                            node.setDirtyCanvas(true, false);
                        };
                        
                        // Initial analysis if there's already text
                        if (textWidget.value) {
                            visualizer.analyzeText(textWidget.value);
                        }
                    }
                    
                }, 10);
                
                return ret;
            };
        }
    }
});