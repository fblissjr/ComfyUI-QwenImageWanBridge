/**
 * Qwen Testing Interface
 * Comprehensive testing environment for Qwen tokens and sequences
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

// Test case templates
const TEST_TEMPLATES = {
    basic_vision: {
        name: "Basic Vision Processing",
        template: "Describe this image: <|vision_start|><|image_pad|><|vision_end|>",
        description: "Simple vision processing without chat formatting",
        category: "vision"
    },
    chat_vision: {
        name: "Chat + Vision",
        template: "<|im_start|>user\nAnalyze this image: <|vision_start|><|image_pad|><|vision_end|>\nWhat objects do you see?<|im_end|>",
        description: "Vision processing within chat format",
        category: "chat"
    },
    spatial_edit: {
        name: "Spatial Editing",
        template: "<|im_start|>user\nEdit the <|object_ref_start|>car<|object_ref_end|> at <|box_start|>100,50,300,200<|box_end|> in this image: <|vision_start|><|image_pad|><|vision_end|>\nMake it red.<|im_end|>",
        description: "Spatial object editing with coordinates",
        category: "spatial"
    },
    polygon_reference: {
        name: "Polygon Reference",
        template: "Modify the building outline <|quad_start|>10,20 100,25 95,80 8,75<|quad_end|> to add windows.",
        description: "Complex polygon-based spatial references",
        category: "spatial"
    },
    multi_object: {
        name: "Multiple Objects",
        template: "Edit the <|object_ref_start|>car<|object_ref_end|> at <|box_start|>100,100,200,200<|box_end|> and the <|object_ref_start|>tree<|object_ref_end|> at <|box_start|>300,50,400,150<|box_end|>",
        description: "Multiple object references in one prompt",
        category: "spatial"
    },
    code_completion: {
        name: "Code Completion",
        template: "<|fim_prefix|>def process_image(image):\n    # Load the image<|fim_suffix|>\n    return processed_image<|fim_middle|>",
        description: "Fill-in-the-middle code completion",
        category: "code"
    },
    tool_calling: {
        name: "Tool Calling",
        template: "<|im_start|>user\nGenerate an image of a sunset<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"generate_image\", \"arguments\": {\"prompt\": \"beautiful sunset over mountains\"}}\n</tool_call><|im_end|>",
        description: "Tool calling within chat format",
        category: "tool"
    }
};

// Validation rules for different token types
const VALIDATION_RULES = {
    box_coordinates: {
        pattern: /^(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)$/,
        validator: (coords) => {
            const [x1, y1, x2, y2] = coords.split(',').map(n => parseFloat(n.trim()));
            return {
                valid: !isNaN(x1) && !isNaN(y1) && !isNaN(x2) && !isNaN(y2) && x2 > x1 && y2 > y1,
                message: "Box coordinates should be: x1,y1,x2,y2 where x2>x1 and y2>y1"
            };
        }
    },
    quad_coordinates: {
        pattern: /^(\d+,\d+\s+){3}\d+,\d+$/,
        validator: (coords) => {
            const pairs = coords.trim().split(/\s+/);
            const valid = pairs.length >= 3 && pairs.every(pair => /^\d+,\d+$/.test(pair));
            return {
                valid: valid,
                message: "Quad coordinates should be space-separated x,y pairs: 'x1,y1 x2,y2 x3,y3 x4,y4'"
            };
        }
    }
};

class QwenTestingInterface {
    constructor() {
        this.currentTest = null;
        this.testResults = [];
        this.customTests = [];
    }
    
    createTestingDialog() {
        const dialog = $el("div", {
            parent: document.body,
            style: {
                position: "fixed",
                top: "10%",
                left: "10%",
                width: "80%",
                height: "80%",
                background: "rgba(20, 20, 20, 0.95)",
                border: "1px solid rgba(255, 255, 255, 0.2)",
                borderRadius: "8px",
                zIndex: 10000,
                overflow: "hidden",
                display: "flex",
                flexDirection: "column",
                backdropFilter: "blur(10px)",
                boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
                fontFamily: "system-ui, -apple-system, sans-serif",
                color: "rgba(255, 255, 255, 0.9)"
            }
        });

        dialog.innerHTML = `
            <div style="
                background: rgba(0, 0, 0, 0.3);
                padding: 15px 20px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <h2 style="margin: 0; color: rgba(255, 255, 255, 0.9); font-size: 18px; font-weight: 500;">Qwen Token Testing Interface</h2>
                <div style="display: flex; gap: 6px;">
                    <button id="exportResults" style="
                        padding: 6px 12px;
                        background: rgba(0, 120, 255, 0.8);
                        color: white;
                        border: 1px solid rgba(0, 120, 255, 0.5);
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 12px;
                    ">Export Results</button>
                    <button id="closeDialog" style="
                        padding: 6px 12px;
                        background: rgba(244, 67, 54, 0.8);
                        color: white;
                        border: 1px solid rgba(244, 67, 54, 0.5);
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 12px;
                    ">Close</button>
                </div>
            </div>
            
            <div style="display: flex; flex: 1; overflow: hidden;">
                <!-- Left Panel: Test Selection -->
                <div style="
                    width: 300px;
                    border-right: 1px solid rgba(255, 255, 255, 0.1);
                    background: rgba(0, 0, 0, 0.2);
                    overflow-y: auto;
                ">
                    <div style="padding: 15px;">
                        <h4 style="margin: 0 0 10px 0; color: rgba(255, 255, 255, 0.9); font-size: 14px; font-weight: 500;">Test Templates</h4>
                        <div id="testCategories">
                            <div class="category-group">
                                <h5 style="margin: 10px 0 5px 0; color: rgba(255, 255, 255, 0.6); font-size: 12px;">Vision</h5>
                                <div id="visionTests"></div>
                            </div>
                            <div class="category-group">
                                <h5 style="margin: 10px 0 5px 0; color: rgba(255, 255, 255, 0.6); font-size: 12px;">Chat</h5>
                                <div id="chatTests"></div>
                            </div>
                            <div class="category-group">
                                <h5 style="margin: 10px 0 5px 0; color: rgba(255, 255, 255, 0.6); font-size: 12px;">Spatial</h5>
                                <div id="spatialTests"></div>
                            </div>
                            <div class="category-group">
                                <h5 style="margin: 10px 0 5px 0; color: rgba(255, 255, 255, 0.6); font-size: 12px;">Code</h5>
                                <div id="codeTests"></div>
                            </div>
                            <div class="category-group">
                                <h5 style="margin: 10px 0 5px 0; color: rgba(255, 255, 255, 0.6); font-size: 12px;">Tool</h5>
                                <div id="toolTests"></div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                            <h5 style="margin: 0 0 10px 0; color: rgba(255, 255, 255, 0.9); font-size: 14px; font-weight: 500;">Custom Test</h5>
                            <button id="createCustomTest" style="
                                width: 100%;
                                padding: 8px;
                                background: rgba(40, 167, 69, 0.8);
                                color: white;
                                border: 1px solid rgba(40, 167, 69, 0.5);
                                border-radius: 4px;
                                cursor: pointer;
                                font-size: 12px;
                            ">Create Custom Test</button>
                        </div>
                    </div>
                </div>
                
                <!-- Right Panel: Test Editor and Results -->
                <div style="flex: 1; display: flex; flex-direction: column; overflow: hidden;">
                    <!-- Test Editor -->
                    <div style="
                        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                        background: rgba(40, 40, 40, 0.3);
                        flex: 1;
                        overflow: hidden;
                        display: flex;
                        flex-direction: column;
                    ">
                        <div style="padding: 15px; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h4 style="margin: 0; color: rgba(255, 255, 255, 0.9); font-size: 14px; font-weight: 500;" id="currentTestName">Select a test template</h4>
                                <div style="display: flex; gap: 6px;">
                                    <button id="validateTest" style="
                                        padding: 6px 12px;
                                        background: rgba(255, 193, 7, 0.8);
                                        color: rgba(33, 37, 41, 0.9);
                                        border: 1px solid rgba(255, 193, 7, 0.5);
                                        border-radius: 4px;
                                        cursor: pointer;
                                        font-size: 12px;
                                    ">Validate</button>
                                    <button id="runTest" style="
                                        padding: 6px 12px;
                                        background: rgba(40, 167, 69, 0.8);
                                        color: white;
                                        border: 1px solid rgba(40, 167, 69, 0.5);
                                        border-radius: 4px;
                                        cursor: pointer;
                                        font-size: 12px;
                                    ">Run Test</button>
                                </div>
                            </div>
                            <p style="margin: 0; color: rgba(255, 255, 255, 0.6); font-size: 12px;" id="testDescription"></p>
                        </div>
                        
                        <div style="flex: 1; padding: 15px; overflow: auto;">
                            <textarea id="testInput" style="
                                width: 100%;
                                height: 200px;
                                font-family: 'Monaco', 'Menlo', monospace;
                                font-size: 12px;
                                background: rgba(255, 255, 255, 0.05);
                                border: 1px solid rgba(255, 255, 255, 0.2);
                                border-radius: 4px;
                                padding: 10px;
                                resize: vertical;
                                color: rgba(255, 255, 255, 0.9);
                            " placeholder="Enter your test text here..."></textarea>
                        </div>
                    </div>
                    
                    <!-- Results Panel -->
                    <div style="
                        height: 300px;
                        border-top: 1px solid rgba(255, 255, 255, 0.1);
                        background: rgba(0, 0, 0, 0.2);
                        overflow: auto;
                    ">
                        <div style="padding: 15px;">
                            <h4 style="margin: 0 0 15px 0; color: rgba(255, 255, 255, 0.9); font-size: 14px; font-weight: 500;">Test Results</h4>
                            <div id="testResults"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.setupEventListeners(dialog);
        this.populateTestTemplates(dialog);
        
        return dialog;
    }
    
    setupEventListeners(dialog) {
        // Close dialog
        dialog.querySelector('#closeDialog').onclick = () => {
            document.body.removeChild(dialog);
        };
        
        // Export results
        dialog.querySelector('#exportResults').onclick = () => {
            this.exportTestResults();
        };
        
        // Validate test
        dialog.querySelector('#validateTest').onclick = () => {
            const input = dialog.querySelector('#testInput').value;
            this.validateTest(input, dialog);
        };
        
        // Run test
        dialog.querySelector('#runTest').onclick = () => {
            const input = dialog.querySelector('#testInput').value;
            this.runTest(input, dialog);
        };
        
        // Create custom test
        dialog.querySelector('#createCustomTest').onclick = () => {
            this.createCustomTest(dialog);
        };
    }
    
    populateTestTemplates(dialog) {
        const categories = {
            vision: dialog.querySelector('#visionTests'),
            chat: dialog.querySelector('#chatTests'),
            spatial: dialog.querySelector('#spatialTests'),
            code: dialog.querySelector('#codeTests'),
            tool: dialog.querySelector('#toolTests')
        };
        
        Object.entries(TEST_TEMPLATES).forEach(([key, test]) => {
            const button = $el("button", {
                style: {
                    width: "100%",
                    padding: "8px 12px",
                    marginBottom: "5px",
                    background: "rgba(255, 255, 255, 0.05)",
                    border: "1px solid rgba(255, 255, 255, 0.2)",
                    borderRadius: "4px",
                    cursor: "pointer",
                    textAlign: "left",
                    fontSize: "12px",
                    color: "rgba(255, 255, 255, 0.9)"
                },
                textContent: test.name
            });
            
            button.onclick = () => {
                this.loadTest(test, dialog);
            };
            
            if (categories[test.category]) {
                categories[test.category].appendChild(button);
            }
        });
    }
    
    loadTest(test, dialog) {
        this.currentTest = test;
        dialog.querySelector('#currentTestName').textContent = test.name;
        dialog.querySelector('#testDescription').textContent = test.description;
        dialog.querySelector('#testInput').value = test.template;
    }
    
    validateTest(input, dialog) {
        const results = {
            timestamp: new Date().toISOString(),
            input: input,
            type: 'validation',
            results: []
        };
        
        // Check for token sequences
        const sequences = this.findTokenSequences(input);
        results.results.push({
            type: 'sequences',
            count: sequences.length,
            details: sequences
        });
        
        // Validate coordinates in spatial tokens
        const coordinateErrors = this.validateCoordinates(input);
        if (coordinateErrors.length > 0) {
            results.results.push({
                type: 'coordinate_errors',
                count: coordinateErrors.length,
                details: coordinateErrors
            });
        }
        
        // Check for unmatched tokens
        const unmatchedTokens = this.findUnmatchedTokens(input);
        if (unmatchedTokens.length > 0) {
            results.results.push({
                type: 'unmatched_tokens',
                count: unmatchedTokens.length,
                details: unmatchedTokens
            });
        }
        
        this.displayResults(results, dialog);
        this.testResults.push(results);
    }
    
    runTest(input, dialog) {
        const results = {
            timestamp: new Date().toISOString(),
            input: input,
            type: 'execution',
            testName: this.currentTest?.name || 'Custom Test',
            results: []
        };
        
        // Simulate token analysis (in real implementation, this would call the actual tokenizer)
        const analysis = this.analyzeTokens(input);
        results.results = analysis;
        
        this.displayResults(results, dialog);
        this.testResults.push(results);
    }
    
    findTokenSequences(text) {
        const sequences = [];
        
        // Vision sequences
        const visionRegex = /<\|vision_start\|>(.*?)<\|vision_end\|>/g;
        let match;
        while ((match = visionRegex.exec(text)) !== null) {
            sequences.push({
                type: 'vision',
                content: match[0],
                inner: match[1],
                position: match.index
            });
        }
        
        // Spatial sequences
        const boxRegex = /<\|box_start\|>(.*?)<\|box_end\|>/g;
        while ((match = boxRegex.exec(text)) !== null) {
            sequences.push({
                type: 'bounding_box',
                content: match[0],
                coordinates: match[1],
                position: match.index
            });
        }
        
        // Object references
        const objRegex = /<\|object_ref_start\|>(.*?)<\|object_ref_end\|>/g;
        while ((match = objRegex.exec(text)) !== null) {
            sequences.push({
                type: 'object_reference',
                content: match[0],
                object: match[1],
                position: match.index
            });
        }
        
        return sequences;
    }
    
    validateCoordinates(text) {
        const errors = [];
        
        // Validate box coordinates
        const boxRegex = /<\|box_start\|>(.*?)<\|box_end\|>/g;
        let match;
        while ((match = boxRegex.exec(text)) !== null) {
            const validation = VALIDATION_RULES.box_coordinates.validator(match[1]);
            if (!validation.valid) {
                errors.push({
                    type: 'box_coordinates',
                    content: match[0],
                    error: validation.message,
                    position: match.index
                });
            }
        }
        
        // Validate quad coordinates
        const quadRegex = /<\|quad_start\|>(.*?)<\|quad_end\|>/g;
        while ((match = quadRegex.exec(text)) !== null) {
            const validation = VALIDATION_RULES.quad_coordinates.validator(match[1]);
            if (!validation.valid) {
                errors.push({
                    type: 'quad_coordinates',
                    content: match[0],
                    error: validation.message,
                    position: match.index
                });
            }
        }
        
        return errors;
    }
    
    findUnmatchedTokens(text) {
        const unmatched = [];
        const tokens = [
            '<|vision_start|>', '<|vision_end|>',
            '<|box_start|>', '<|box_end|>',
            '<|quad_start|>', '<|quad_end|>',
            '<|object_ref_start|>', '<|object_ref_end|>',
            '<|im_start|>', '<|im_end|>'
        ];
        
        const pairs = [
            ['<|vision_start|>', '<|vision_end|>'],
            ['<|box_start|>', '<|box_end|>'],
            ['<|quad_start|>', '<|quad_end|>'],
            ['<|object_ref_start|>', '<|object_ref_end|>'],
            ['<|im_start|>', '<|im_end|>']
        ];
        
        pairs.forEach(([start, end]) => {
            const startCount = (text.match(new RegExp(this.escapeRegex(start), 'g')) || []).length;
            const endCount = (text.match(new RegExp(this.escapeRegex(end), 'g')) || []).length;
            
            if (startCount !== endCount) {
                unmatched.push({
                    startToken: start,
                    endToken: end,
                    startCount: startCount,
                    endCount: endCount
                });
            }
        });
        
        return unmatched;
    }
    
    analyzeTokens(text) {
        // Simplified token analysis (would use actual tokenizer in production)
        const specialTokens = [
            '<|vision_start|>', '<|vision_end|>', '<|image_pad|>', '<|video_pad|>', '<|vision_pad|>',
            '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>',
            '<|quad_start|>', '<|quad_end|>', '<|im_start|>', '<|im_end|>', '<|endoftext|>',
            '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>', '<|fim_pad|>',
            '<|repo_name|>', '<|file_sep|>', '<tool_call>', '</tool_call>'
        ];
        
        const analysis = {
            totalLength: text.length,
            estimatedTokens: text.split(/\s+/).length,
            specialTokensFound: [],
            sequences: this.findTokenSequences(text),
            efficiency: 0
        };
        
        specialTokens.forEach(token => {
            const count = (text.match(new RegExp(this.escapeRegex(token), 'g')) || []).length;
            if (count > 0) {
                analysis.specialTokensFound.push({
                    token: token,
                    count: count
                });
            }
        });
        
        analysis.efficiency = analysis.specialTokensFound.length / analysis.estimatedTokens;
        
        return [
            {
                type: 'token_analysis',
                details: analysis
            }
        ];
    }
    
    displayResults(results, dialog) {
        const resultsDiv = dialog.querySelector('#testResults');
        
        const resultElement = $el("div", {
            style: {
                marginBottom: "15px",
                padding: "12px",
                background: "rgba(255, 255, 255, 0.05)",
                border: "1px solid rgba(255, 255, 255, 0.1)",
                borderRadius: "4px"
            }
        });
        
        const timestamp = new Date(results.timestamp).toLocaleTimeString();
        const typeColor = results.type === 'validation' ? '#ffc107' : '#007bff';
        
        resultElement.innerHTML = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <strong style="color: rgba(255, 255, 255, 0.9);">${results.testName || 'Test'}</strong>
                <span style="
                    background: ${typeColor}; 
                    color: white; 
                    padding: 2px 8px; 
                    border-radius: 12px; 
                    font-size: 11px;
                ">${results.type}</span>
            </div>
            <div style="font-size: 12px; color: rgba(255, 255, 255, 0.6); margin-bottom: 8px;">${timestamp}</div>
            <div style="font-size: 13px; color: rgba(255, 255, 255, 0.8);">
                ${results.results.map(result => this.formatResult(result)).join('<br>')}
            </div>
        `;
        
        resultsDiv.appendChild(resultElement);
        resultsDiv.scrollTop = resultsDiv.scrollHeight;
    }
    
    formatResult(result) {
        switch (result.type) {
            case 'sequences':
                return `<strong>Sequences:</strong> ${result.count} found (${result.details.map(s => s.type).join(', ')})`;
            case 'coordinate_errors':
                return `<strong>Coordinate Errors:</strong> ${result.count} found`;
            case 'unmatched_tokens':
                return `<strong>Unmatched Tokens:</strong> ${result.count} pairs unmatched`;
            case 'token_analysis':
                const analysis = result.details;
                return `<strong>Analysis:</strong> ~${analysis.estimatedTokens} tokens, ${analysis.specialTokensFound.length} special tokens, ${(analysis.efficiency * 100).toFixed(1)}% efficiency`;
            default:
                return `<strong>${result.type}:</strong> ${JSON.stringify(result.details)}`;
        }
    }
    
    createCustomTest(dialog) {
        const customTestDialog = $el("div", {
            parent: document.body,
            style: {
                position: "fixed",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                width: "500px",
                background: "rgba(20, 20, 20, 0.95)",
                border: "1px solid rgba(255, 255, 255, 0.2)",
                borderRadius: "8px",
                padding: "20px",
                zIndex: 10001,
                backdropFilter: "blur(10px)",
                boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
                fontFamily: "system-ui, -apple-system, sans-serif",
                color: "rgba(255, 255, 255, 0.9)"
            }
        });
        
        customTestDialog.innerHTML = `
            <h3 style="margin: 0 0 15px 0; color: rgba(255, 255, 255, 0.9); font-size: 18px; font-weight: 500;">Create Custom Test</h3>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; color: rgba(255, 255, 255, 0.9); font-size: 14px;">Test Name:</label>
                <input type="text" id="customTestName" style="
                    width: 100%; 
                    padding: 8px; 
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.2); 
                    border-radius: 4px;
                    color: rgba(255, 255, 255, 0.9);
                    font-size: 12px;
                ">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; color: rgba(255, 255, 255, 0.9); font-size: 14px;">Description:</label>
                <input type="text" id="customTestDesc" style="
                    width: 100%; 
                    padding: 8px; 
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.2); 
                    border-radius: 4px;
                    color: rgba(255, 255, 255, 0.9);
                    font-size: 12px;
                ">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; color: rgba(255, 255, 255, 0.9); font-size: 14px;">Template:</label>
                <textarea id="customTestTemplate" rows="6" style="
                    width: 100%; 
                    padding: 8px; 
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.2); 
                    border-radius: 4px; 
                    font-family: 'Monaco', 'Menlo', monospace; 
                    font-size: 11px;
                    color: rgba(255, 255, 255, 0.9);
                    resize: vertical;
                "></textarea>
            </div>
            <div style="display: flex; gap: 6px; justify-content: flex-end;">
                <button id="saveCustomTest" style="
                    padding: 6px 12px; 
                    background: rgba(40, 167, 69, 0.8); 
                    color: white; 
                    border: 1px solid rgba(40, 167, 69, 0.5); 
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                ">Save Test</button>
                <button id="cancelCustomTest" style="
                    padding: 6px 12px; 
                    background: rgba(108, 117, 125, 0.8); 
                    color: white; 
                    border: 1px solid rgba(108, 117, 125, 0.5); 
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                ">Cancel</button>
            </div>
        `;
        
        customTestDialog.querySelector('#saveCustomTest').onclick = () => {
            const name = customTestDialog.querySelector('#customTestName').value;
            const description = customTestDialog.querySelector('#customTestDesc').value;
            const template = customTestDialog.querySelector('#customTestTemplate').value;
            
            if (name && template) {
                const customTest = {
                    name: name,
                    description: description,
                    template: template,
                    category: 'custom'
                };
                
                this.customTests.push(customTest);
                this.loadTest(customTest, dialog);
                document.body.removeChild(customTestDialog);
            } else {
                alert('Please provide at least a name and template.');
            }
        };
        
        customTestDialog.querySelector('#cancelCustomTest').onclick = () => {
            document.body.removeChild(customTestDialog);
        };
    }
    
    exportTestResults() {
        const exportData = {
            timestamp: new Date().toISOString(),
            totalTests: this.testResults.length,
            customTests: this.customTests,
            results: this.testResults
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `qwen_test_results_${new Date().toISOString().slice(0, 19)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
}

// Add testing interface to Qwen nodes
app.registerExtension({
    name: "Comfy.QwenTestingInterface",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "QwenVLTextEncoder" || nodeData.name === "QwenTokenDebugger") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const node = this;
                
                // Add testing interface button
                setTimeout(() => {
                    const testBtn = node.addWidget("button", "Open Testing Interface", "test", () => {
                        const testInterface = new QwenTestingInterface();
                        testInterface.createTestingDialog();
                    });
                    
                }, 10);
                
                return ret;
            };
        }
    }
});