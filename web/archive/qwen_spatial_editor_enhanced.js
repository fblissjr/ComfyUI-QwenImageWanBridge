/**
 * Enhanced Qwen Spatial Editor
 * Interactive image-based spatial coordinate editor for ComfyUI
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

class QwenInteractiveSpatialEditor {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.imageCanvas = null;
        this.imageCtx = null;
        this.regions = [];
        this.currentRegion = null;
        this.isDrawing = false;
        this.drawingMode = "bounding_box"; // "bounding_box", "polygon", "object_reference"
        this.currentImage = null;
        this.imageScale = 1;
        this.imageOffset = { x: 0, y: 0 };
        this.polygonPoints = [];
        this.debugMode = true;
        
        this.colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", 
            "#FF00FF", "#00FFFF", "#FFA500", "#800080"
        ];
    }
    
    createSpatialEditorDialog(node = null) {
        // Create main dialog
        const dialog = $el("div", {
            parent: document.body,
            style: {
                position: "fixed",
                top: "5%",
                left: "5%",
                width: "90%",
                height: "90%",
                background: "white",
                border: "2px solid #333",
                borderRadius: "8px",
                zIndex: 10000,
                overflow: "hidden",
                display: "flex",
                flexDirection: "column",
                boxShadow: "0 4px 20px rgba(0,0,0,0.5)"
            }
        });

        dialog.innerHTML = `
            <!-- Header -->
            <div style="
                background: #2c3e50;
                color: white;
                padding: 15px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #34495e;
            ">
                <h2 style="margin: 0;">Qwen Interactive Spatial Editor</h2>
                <div>
                    <span id="debugToggle" style="
                        margin-right: 15px;
                        padding: 5px 10px;
                        background: #27ae60;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 12px;
                    ">DEBUG: ON</span>
                    <button id="closeEditor" style="
                        padding: 8px 16px;
                        background: #e74c3c;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    ">Close</button>
                </div>
            </div>
            
            <!-- Main Content -->
            <div style="flex: 1; display: flex; overflow: hidden;">
                <!-- Left Panel: Tools -->
                <div style="
                    width: 300px;
                    background: #ecf0f1;
                    border-right: 1px solid #bdc3c7;
                    overflow-y: auto;
                    padding: 15px;
                ">
                    <!-- Image Upload -->
                    <div style="margin-bottom: 20px;">
                        <h4 style="margin: 0 0 10px 0;">1. Load Image</h4>
                        <input type="file" id="imageUpload" accept="image/*" style="
                            width: 100%;
                            padding: 8px;
                            border: 1px solid #bdc3c7;
                            border-radius: 4px;
                            margin-bottom: 10px;
                        ">
                        <div style="font-size: 12px; color: #7f8c8d;">
                            Or load from connected node input
                        </div>
                        <button id="loadFromNode" style="
                            width: 100%;
                            padding: 8px;
                            background: #3498db;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            margin-top: 5px;
                        ">Load from Node</button>
                    </div>
                    
                    <!-- Drawing Tools -->
                    <div style="margin-bottom: 20px;">
                        <h4 style="margin: 0 0 10px 0;">2. Drawing Mode</h4>
                        <select id="drawingMode" style="
                            width: 100%;
                            padding: 8px;
                            border: 1px solid #bdc3c7;
                            border-radius: 4px;
                            margin-bottom: 10px;
                        ">
                            <option value="bounding_box">Bounding Box</option>
                            <option value="polygon">Polygon</option>
                            <option value="object_reference">Object Point</option>
                        </select>
                        
                        <div style="margin-bottom: 10px;">
                            <label style="display: block; margin-bottom: 5px; font-weight: bold;">Region Label:</label>
                            <input type="text" id="regionLabel" value="object" style="
                                width: 100%;
                                padding: 6px;
                                border: 1px solid #bdc3c7;
                                border-radius: 4px;
                            ">
                        </div>
                        
                        <div id="drawingInstructions" style="
                            padding: 10px;
                            background: #d5dbdb;
                            border-radius: 4px;
                            font-size: 12px;
                            color: #2c3e50;
                        ">
                            Select bounding box mode and click-drag on the image to create regions.
                        </div>
                    </div>
                    
                    <!-- Region List -->
                    <div style="margin-bottom: 20px;">
                        <h4 style="margin: 0 0 10px 0;">3. Regions</h4>
                        <div id="regionsList" style="
                            max-height: 200px;
                            overflow-y: auto;
                            border: 1px solid #bdc3c7;
                            border-radius: 4px;
                            padding: 5px;
                            background: white;
                        ">
                            <div style="color: #7f8c8d; text-align: center; padding: 20px; font-style: italic;">
                                No regions created yet
                            </div>
                        </div>
                        
                        <div style="margin-top: 10px;">
                            <button id="clearRegions" style="
                                width: 48%;
                                padding: 6px;
                                background: #e67e22;
                                color: white;
                                border: none;
                                border-radius: 4px;
                                margin-right: 4%;
                            ">Clear All</button>
                            <button id="deleteSelected" style="
                                width: 48%;
                                padding: 6px;
                                background: #e74c3c;
                                color: white;
                                border: none;
                                border-radius: 4px;
                            ">Delete Selected</button>
                        </div>
                    </div>
                    
                    <!-- Token Generation -->
                    <div>
                        <h4 style="margin: 0 0 10px 0;">4. Generate Tokens</h4>
                        <div style="margin-bottom: 10px;">
                            <label style="display: block; margin-bottom: 5px;">Instruction:</label>
                            <textarea id="instructionText" rows="3" style="
                                width: 100%;
                                padding: 6px;
                                border: 1px solid #bdc3c7;
                                border-radius: 4px;
                                font-size: 12px;
                            " placeholder="What should be done with these regions?">Edit these objects</textarea>
                        </div>
                        
                        <div style="margin-bottom: 10px;">
                            <label style="display: flex; align-items: center; margin-bottom: 5px;">
                                <input type="checkbox" id="includeChatFormat" checked style="margin-right: 8px;">
                                Include chat format
                            </label>
                            <label style="display: flex; align-items: center;">
                                <input type="checkbox" id="includeVisionTokens" checked style="margin-right: 8px;">
                                Include vision tokens
                            </label>
                        </div>
                        
                        <button id="generateTokens" style="
                            width: 100%;
                            padding: 10px;
                            background: #27ae60;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            font-weight: bold;
                        ">Generate Spatial Tokens</button>
                    </div>
                </div>
                
                <!-- Center Panel: Canvas -->
                <div style="
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    background: #34495e;
                    position: relative;
                ">
                    <!-- Canvas Controls -->
                    <div style="
                        background: #2c3e50;
                        padding: 10px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        gap: 15px;
                        color: white;
                    ">
                        <button id="zoomIn" style="
                            padding: 5px 10px;
                            background: #3498db;
                            color: white;
                            border: none;
                            border-radius: 4px;
                        ">Zoom +</button>
                        <span id="zoomLevel">100%</span>
                        <button id="zoomOut" style="
                            padding: 5px 10px;
                            background: #3498db;
                            color: white;
                            border: none;
                            border-radius: 4px;
                        ">Zoom -</button>
                        <button id="resetZoom" style="
                            padding: 5px 10px;
                            background: #95a5a6;
                            color: white;
                            border: none;
                            border-radius: 4px;
                        ">Reset</button>
                        <div style="margin-left: 20px;">
                            <span id="mouseCoords">Mouse: (0, 0)</span>
                        </div>
                    </div>
                    
                    <!-- Drawing Canvas -->
                    <div id="canvasContainer" style="
                        flex: 1;
                        overflow: hidden;
                        position: relative;
                        cursor: crosshair;
                    ">
                        <canvas id="drawingCanvas" style="
                            position: absolute;
                            top: 0;
                            left: 0;
                            z-index: 2;
                        "></canvas>
                        <canvas id="imageCanvas" style="
                            position: absolute;
                            top: 0;
                            left: 0;
                            z-index: 1;
                        "></canvas>
                    </div>
                </div>
                
                <!-- Right Panel: Output -->
                <div style="
                    width: 350px;
                    background: #ecf0f1;
                    border-left: 1px solid #bdc3c7;
                    overflow-y: auto;
                    padding: 15px;
                ">
                    <h4 style="margin: 0 0 15px 0;">Generated Output</h4>
                    
                    <!-- Spatial Tokens -->
                    <div style="margin-bottom: 20px;">
                        <label style="display: block; margin-bottom: 5px; font-weight: bold;">Spatial Tokens:</label>
                        <textarea id="spatialTokensOutput" rows="4" readonly style="
                            width: 100%;
                            padding: 8px;
                            border: 1px solid #bdc3c7;
                            border-radius: 4px;
                            font-family: monospace;
                            font-size: 11px;
                            background: #f8f9fa;
                        " placeholder="Generated spatial tokens will appear here..."></textarea>
                        <button id="copySpatialTokens" style="
                            width: 100%;
                            padding: 6px;
                            background: #3498db;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            margin-top: 5px;
                            font-size: 12px;
                        ">Copy Spatial Tokens</button>
                    </div>
                    
                    <!-- Full Prompt -->
                    <div style="margin-bottom: 20px;">
                        <label style="display: block; margin-bottom: 5px; font-weight: bold;">Full Prompt:</label>
                        <textarea id="fullPromptOutput" rows="6" readonly style="
                            width: 100%;
                            padding: 8px;
                            border: 1px solid #bdc3c7;
                            border-radius: 4px;
                            font-family: monospace;
                            font-size: 11px;
                            background: #f8f9fa;
                        " placeholder="Generated full prompt will appear here..."></textarea>
                        <button id="copyFullPrompt" style="
                            width: 100%;
                            padding: 6px;
                            background: #27ae60;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            margin-top: 5px;
                            font-size: 12px;
                        ">Copy Full Prompt</button>
                    </div>
                    
                    <!-- Send to Node -->
                    <div style="margin-bottom: 20px;">
                        <button id="sendToNode" style="
                            width: 100%;
                            padding: 10px;
                            background: #8e44ad;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            font-weight: bold;
                        ">Send to Connected Node</button>
                    </div>
                    
                    <!-- Debug Panel -->
                    <div id="debugPanel" style="
                        background: #2c3e50;
                        color: white;
                        padding: 10px;
                        border-radius: 4px;
                        margin-bottom: 20px;
                    ">
                        <h5 style="margin: 0 0 10px 0;">Debug Info:</h5>
                        <div id="debugContent" style="
                            font-family: monospace;
                            font-size: 10px;
                            white-space: pre-wrap;
                            max-height: 200px;
                            overflow-y: auto;
                        ">Ready for image input...</div>
                    </div>
                    
                    <!-- Export Options -->
                    <div>
                        <h5 style="margin: 0 0 10px 0;">Export:</h5>
                        <button id="exportJSON" style="
                            width: 100%;
                            padding: 8px;
                            background: #34495e;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            margin-bottom: 5px;
                        ">Export as JSON</button>
                        <button id="exportAnnotatedImage" style="
                            width: 100%;
                            padding: 8px;
                            background: #16a085;
                            color: white;
                            border: none;
                            border-radius: 4px;
                        ">Export Annotated Image</button>
                    </div>
                </div>
            </div>
        `;

        this.setupEventListeners(dialog, node);
        this.setupCanvas(dialog);
        this.updateDebugPanel("Spatial editor initialized. Load an image to begin.");
        
        return dialog;
    }
    
    setupEventListeners(dialog, node) {
        // Close button
        dialog.querySelector('#closeEditor').onclick = () => {
            document.body.removeChild(dialog);
        };
        
        // Debug toggle
        dialog.querySelector('#debugToggle').onclick = (e) => {
            this.debugMode = !this.debugMode;
            e.target.textContent = `DEBUG: ${this.debugMode ? 'ON' : 'OFF'}`;
            e.target.style.background = this.debugMode ? '#27ae60' : '#95a5a6';
            dialog.querySelector('#debugPanel').style.display = this.debugMode ? 'block' : 'none';
        };
        
        // Image upload
        dialog.querySelector('#imageUpload').onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                this.loadImageFromFile(file, dialog);
            }
        };
        
        // Load from node
        dialog.querySelector('#loadFromNode').onclick = () => {
            this.loadImageFromNode(node, dialog);
        };
        
        // Drawing mode change
        dialog.querySelector('#drawingMode').onchange = (e) => {
            this.drawingMode = e.target.value;
            this.updateDrawingInstructions(dialog);
            this.polygonPoints = []; // Reset polygon points
            this.updateDebugPanel(`Drawing mode changed to: ${this.drawingMode}`);
        };
        
        // Region management
        dialog.querySelector('#clearRegions').onclick = () => {
            this.regions = [];
            this.updateRegionsList(dialog);
            this.redrawCanvas();
            this.updateDebugPanel("All regions cleared");
        };
        
        dialog.querySelector('#deleteSelected').onclick = () => {
            // Find selected region and remove
            const selectedIndex = this.getSelectedRegionIndex();
            if (selectedIndex >= 0) {
                const removed = this.regions.splice(selectedIndex, 1)[0];
                this.updateRegionsList(dialog);
                this.redrawCanvas();
                this.updateDebugPanel(`Deleted region: ${removed.label}`);
            }
        };
        
        // Generate tokens
        dialog.querySelector('#generateTokens').onclick = () => {
            this.generateTokens(dialog);
        };
        
        // Copy buttons
        dialog.querySelector('#copySpatialTokens').onclick = () => {
            const tokens = dialog.querySelector('#spatialTokensOutput').value;
            navigator.clipboard.writeText(tokens);
            this.updateDebugPanel("Spatial tokens copied to clipboard");
        };
        
        dialog.querySelector('#copyFullPrompt').onclick = () => {
            const prompt = dialog.querySelector('#fullPromptOutput').value;
            navigator.clipboard.writeText(prompt);
            this.updateDebugPanel("Full prompt copied to clipboard");
        };
        
        // Send to node
        dialog.querySelector('#sendToNode').onclick = () => {
            this.sendToNode(node, dialog);
        };
        
        // Export functions
        dialog.querySelector('#exportJSON').onclick = () => {
            this.exportJSON(dialog);
        };
        
        dialog.querySelector('#exportAnnotatedImage').onclick = () => {
            this.exportAnnotatedImage(dialog);
        };
        
        // Zoom controls
        dialog.querySelector('#zoomIn').onclick = () => {
            this.imageScale = Math.min(this.imageScale * 1.2, 5);
            this.updateCanvas(dialog);
        };
        
        dialog.querySelector('#zoomOut').onclick = () => {
            this.imageScale = Math.max(this.imageScale / 1.2, 0.1);
            this.updateCanvas(dialog);
        };
        
        dialog.querySelector('#resetZoom').onclick = () => {
            this.imageScale = 1;
            this.imageOffset = { x: 0, y: 0 };
            this.updateCanvas(dialog);
        };
    }
    
    setupCanvas(dialog) {
        const container = dialog.querySelector('#canvasContainer');
        this.canvas = dialog.querySelector('#drawingCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.imageCanvas = dialog.querySelector('#imageCanvas');
        this.imageCtx = this.imageCanvas.getContext('2d');
        
        // Set initial canvas size
        this.resizeCanvases(container.clientWidth, container.clientHeight);
        
        // Mouse event handlers
        this.setupCanvasEvents(dialog);
        
        // Resize observer
        const resizeObserver = new ResizeObserver(() => {
            this.resizeCanvases(container.clientWidth, container.clientHeight);
            this.updateCanvas(dialog);
        });
        resizeObserver.observe(container);
    }
    
    setupCanvasEvents(dialog) {
        let startX, startY;
        let isDragging = false;
        
        // Mouse down
        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            startX = (e.clientX - rect.left) / this.imageScale - this.imageOffset.x;
            startY = (e.clientY - rect.top) / this.imageScale - this.imageOffset.y;
            
            if (this.drawingMode === 'polygon') {
                // Add point to polygon
                this.polygonPoints.push([Math.round(startX), Math.round(startY)]);
                this.updateDebugPanel(`Polygon point ${this.polygonPoints.length}: (${Math.round(startX)}, ${Math.round(startY)})`);
                this.redrawCanvas();
                
                // Double click to finish polygon
                if (e.detail === 2 && this.polygonPoints.length >= 3) {
                    this.finishPolygon(dialog);
                }
            } else {
                isDragging = true;
                this.isDrawing = true;
            }
        });
        
        // Mouse move
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const currentX = (e.clientX - rect.left) / this.imageScale - this.imageOffset.x;
            const currentY = (e.clientY - rect.top) / this.imageScale - this.imageOffset.y;
            
            // Update mouse coordinates display
            dialog.querySelector('#mouseCoords').textContent = 
                `Mouse: (${Math.round(currentX)}, ${Math.round(currentY)})`;
            
            if (isDragging && this.drawingMode === 'bounding_box') {
                this.redrawCanvas();
                this.drawPreviewBox(startX, startY, currentX, currentY);
            }
        });
        
        // Mouse up
        this.canvas.addEventListener('mouseup', (e) => {
            if (!isDragging) return;
            
            const rect = this.canvas.getBoundingClientRect();
            const endX = (e.clientX - rect.left) / this.imageScale - this.imageOffset.x;
            const endY = (e.clientY - rect.top) / this.imageScale - this.imageOffset.y;
            
            if (this.drawingMode === 'bounding_box') {
                const minSize = 5; // Minimum box size
                if (Math.abs(endX - startX) > minSize && Math.abs(endY - startY) > minSize) {
                    this.addBoundingBox(startX, startY, endX, endY, dialog);
                }
            } else if (this.drawingMode === 'object_reference') {
                this.addObjectReference(startX, startY, dialog);
            }
            
            isDragging = false;
            this.isDrawing = false;
        });
        
        // Keyboard events for polygon finishing
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && this.drawingMode === 'polygon' && this.polygonPoints.length >= 3) {
                this.finishPolygon(dialog);
            } else if (e.key === 'Escape') {
                this.polygonPoints = [];
                this.redrawCanvas();
                this.updateDebugPanel("Polygon drawing cancelled");
            }
        });
    }
    
    updateDrawingInstructions(dialog) {
        const instructions = {
            'bounding_box': 'Click and drag to create bounding boxes around objects.',
            'polygon': 'Click to add points. Double-click or press Enter to finish polygon.',
            'object_reference': 'Click on objects to mark them with reference points.'
        };
        
        dialog.querySelector('#drawingInstructions').textContent = 
            instructions[this.drawingMode] || 'Select a drawing mode.';
    }
    
    loadImageFromFile(file, dialog) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                this.imageScale = 1;
                this.imageOffset = { x: 0, y: 0 };
                this.regions = [];
                this.updateCanvas(dialog);
                this.updateRegionsList(dialog);
                this.updateDebugPanel(`Image loaded: ${img.width}x${img.height}px`);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    loadImageFromNode(node, dialog) {
        // This would interface with the node's image input
        this.updateDebugPanel("Loading image from connected node... (Feature in development)");
        
        // For now, show a placeholder
        if (!this.currentImage) {
            // Create a placeholder image
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 512;
            const ctx = canvas.getContext('2d');
            
            // Draw placeholder
            ctx.fillStyle = '#ecf0f1';
            ctx.fillRect(0, 0, 512, 512);
            ctx.fillStyle = '#95a5a6';
            ctx.font = '24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Load an image', 256, 230);
            ctx.fillText('to begin editing', 256, 260);
            ctx.fillText('spatial regions', 256, 290);
            
            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                this.updateCanvas(dialog);
                this.updateDebugPanel("Placeholder image created. Upload a real image to continue.");
            };
            img.src = canvas.toDataURL();
        }
    }
    
    updateCanvas(dialog) {
        if (!this.currentImage) return;
        
        this.imageCtx.clearRect(0, 0, this.imageCanvas.width, this.imageCanvas.height);
        
        // Draw image
        const imgWidth = this.currentImage.width * this.imageScale;
        const imgHeight = this.currentImage.height * this.imageScale;
        const x = this.imageOffset.x * this.imageScale;
        const y = this.imageOffset.y * this.imageScale;
        
        this.imageCtx.drawImage(this.currentImage, x, y, imgWidth, imgHeight);
        
        // Update zoom display
        dialog.querySelector('#zoomLevel').textContent = `${Math.round(this.imageScale * 100)}%`;
        
        // Redraw regions
        this.redrawCanvas();
    }
    
    redrawCanvas() {
        if (!this.canvas) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw existing regions
        this.regions.forEach((region, index) => {
            this.drawRegion(region, index);
        });
        
        // Draw polygon in progress
        if (this.drawingMode === 'polygon' && this.polygonPoints.length > 0) {
            this.drawPolygonInProgress();
        }
    }
    
    drawRegion(region, index) {
        const color = this.colors[index % this.colors.length];
        this.ctx.strokeStyle = color;
        this.ctx.fillStyle = color + '40'; // Semi-transparent
        this.ctx.lineWidth = 2;
        
        if (region.type === 'bounding_box') {
            const [x1, y1, x2, y2] = region.coords;
            const drawX1 = (x1 + this.imageOffset.x) * this.imageScale;
            const drawY1 = (y1 + this.imageOffset.y) * this.imageScale;
            const drawX2 = (x2 + this.imageOffset.x) * this.imageScale;
            const drawY2 = (y2 + this.imageOffset.y) * this.imageScale;
            
            this.ctx.strokeRect(drawX1, drawY1, drawX2 - drawX1, drawY2 - drawY1);
            this.ctx.fillRect(drawX1, drawY1, drawX2 - drawX1, drawY2 - drawY1);
            
            // Draw label
            this.ctx.fillStyle = color;
            this.ctx.font = '14px Arial';
            this.ctx.fillText(`${index + 1}. ${region.label}`, drawX1, drawY1 - 5);
        } else if (region.type === 'polygon') {
            const points = region.coords.map(([x, y]) => [
                (x + this.imageOffset.x) * this.imageScale,
                (y + this.imageOffset.y) * this.imageScale
            ]);
            
            if (points.length >= 3) {
                this.ctx.beginPath();
                this.ctx.moveTo(points[0][0], points[0][1]);
                for (let i = 1; i < points.length; i++) {
                    this.ctx.lineTo(points[i][0], points[i][1]);
                }
                this.ctx.closePath();
                this.ctx.stroke();
                this.ctx.fill();
                
                // Draw points
                points.forEach(([x, y], i) => {
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, 4, 0, 2 * Math.PI);
                    this.ctx.fillStyle = color;
                    this.ctx.fill();
                    
                    // Point number
                    this.ctx.fillStyle = 'white';
                    this.ctx.font = '10px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.fillText(i + 1, x, y + 3);
                });
                
                // Label
                this.ctx.fillStyle = color;
                this.ctx.font = '14px Arial';
                this.ctx.textAlign = 'left';
                this.ctx.fillText(`${index + 1}. ${region.label}`, points[0][0], points[0][1] - 10);
            }
        } else if (region.type === 'object_reference') {
            const [x, y] = region.coords;
            const drawX = (x + this.imageOffset.x) * this.imageScale;
            const drawY = (y + this.imageOffset.y) * this.imageScale;
            
            // Draw crosshair
            this.ctx.beginPath();
            this.ctx.moveTo(drawX - 10, drawY);
            this.ctx.lineTo(drawX + 10, drawY);
            this.ctx.moveTo(drawX, drawY - 10);
            this.ctx.lineTo(drawX, drawY + 10);
            this.ctx.stroke();
            
            // Draw circle
            this.ctx.beginPath();
            this.ctx.arc(drawX, drawY, 8, 0, 2 * Math.PI);
            this.ctx.stroke();
            
            // Label
            this.ctx.fillStyle = color;
            this.ctx.font = '14px Arial';
            this.ctx.fillText(`${index + 1}. ${region.label}`, drawX + 15, drawY - 5);
        }
    }
    
    drawPreviewBox(x1, y1, x2, y2) {
        const drawX1 = (x1 + this.imageOffset.x) * this.imageScale;
        const drawY1 = (y1 + this.imageOffset.y) * this.imageScale;
        const drawX2 = (x2 + this.imageOffset.x) * this.imageScale;
        const drawY2 = (y2 + this.imageOffset.y) * this.imageScale;
        
        this.ctx.strokeStyle = '#3498db';
        this.ctx.setLineDash([5, 5]);
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(drawX1, drawY1, drawX2 - drawX1, drawY2 - drawY1);
        this.ctx.setLineDash([]);
    }
    
    drawPolygonInProgress() {
        if (this.polygonPoints.length === 0) return;
        
        const points = this.polygonPoints.map(([x, y]) => [
            (x + this.imageOffset.x) * this.imageScale,
            (y + this.imageOffset.y) * this.imageScale
        ]);
        
        this.ctx.strokeStyle = '#3498db';
        this.ctx.setLineDash([5, 5]);
        this.ctx.lineWidth = 2;
        
        // Draw lines between points
        if (points.length > 1) {
            this.ctx.beginPath();
            this.ctx.moveTo(points[0][0], points[0][1]);
            for (let i = 1; i < points.length; i++) {
                this.ctx.lineTo(points[i][0], points[i][1]);
            }
            this.ctx.stroke();
        }
        
        // Draw points
        this.ctx.setLineDash([]);
        points.forEach(([x, y], i) => {
            this.ctx.beginPath();
            this.ctx.arc(x, y, 4, 0, 2 * Math.PI);
            this.ctx.fillStyle = '#3498db';
            this.ctx.fill();
            
            this.ctx.fillStyle = 'white';
            this.ctx.font = '10px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(i + 1, x, y + 3);
        });
    }
    
    addBoundingBox(x1, y1, x2, y2, dialog) {
        const label = dialog.querySelector('#regionLabel').value || `box_${this.regions.length + 1}`;
        
        const region = {
            type: 'bounding_box',
            label: label,
            coords: [
                Math.round(Math.min(x1, x2)),
                Math.round(Math.min(y1, y2)),
                Math.round(Math.max(x1, x2)),
                Math.round(Math.max(y1, y2))
            ]
        };
        
        this.regions.push(region);
        this.updateRegionsList(dialog);
        this.redrawCanvas();
        this.updateDebugPanel(`Added bounding box: ${region.label} at [${region.coords.join(',')}]`);
    }
    
    addObjectReference(x, y, dialog) {
        const label = dialog.querySelector('#regionLabel').value || `object_${this.regions.length + 1}`;
        
        const region = {
            type: 'object_reference',
            label: label,
            coords: [Math.round(x), Math.round(y)]
        };
        
        this.regions.push(region);
        this.updateRegionsList(dialog);
        this.redrawCanvas();
        this.updateDebugPanel(`Added object reference: ${region.label} at (${region.coords.join(',')})`);
    }
    
    finishPolygon(dialog) {
        if (this.polygonPoints.length < 3) {
            this.updateDebugPanel("Polygon needs at least 3 points");
            return;
        }
        
        const label = dialog.querySelector('#regionLabel').value || `polygon_${this.regions.length + 1}`;
        
        const region = {
            type: 'polygon',
            label: label,
            coords: [...this.polygonPoints] // Copy the points
        };
        
        this.regions.push(region);
        this.polygonPoints = [];
        this.updateRegionsList(dialog);
        this.redrawCanvas();
        this.updateDebugPanel(`Added polygon: ${region.label} with ${region.coords.length} points`);
    }
    
    updateRegionsList(dialog) {
        const listContainer = dialog.querySelector('#regionsList');
        
        if (this.regions.length === 0) {
            listContainer.innerHTML = `
                <div style="color: #7f8c8d; text-align: center; padding: 20px; font-style: italic;">
                    No regions created yet
                </div>
            `;
            return;
        }
        
        const listHTML = this.regions.map((region, index) => {
            const color = this.colors[index % this.colors.length];
            let coordsDisplay = '';
            
            if (region.type === 'bounding_box') {
                coordsDisplay = `[${region.coords.join(',')}]`;
            } else if (region.type === 'polygon') {
                coordsDisplay = `${region.coords.length} points`;
            } else if (region.type === 'object_reference') {
                coordsDisplay = `(${region.coords.join(',')})`;
            }
            
            return `
                <div style="
                    padding: 8px;
                    margin-bottom: 5px;
                    border: 1px solid ${color};
                    border-radius: 4px;
                    background: ${color}20;
                    cursor: pointer;
                " data-region-index="${index}">
                    <div style="font-weight: bold; color: ${color};">
                        ${index + 1}. ${region.label}
                    </div>
                    <div style="font-size: 12px; color: #2c3e50;">
                        ${region.type}: ${coordsDisplay}
                    </div>
                </div>
            `;
        }).join('');
        
        listContainer.innerHTML = listHTML;
        
        // Add click handlers for region selection
        listContainer.querySelectorAll('[data-region-index]').forEach(item => {
            item.addEventListener('click', (e) => {
                // Remove previous selection
                listContainer.querySelectorAll('[data-region-index]').forEach(el => {
                    el.style.background = el.style.background.replace('60', '20');
                });
                
                // Add selection
                item.style.background = item.style.background.replace('20', '60');
            });
        });
    }
    
    generateTokens(dialog) {
        if (this.regions.length === 0) {
            this.updateDebugPanel("No regions to generate tokens for");
            return;
        }
        
        const instruction = dialog.querySelector('#instructionText').value || "Edit these regions";
        const includeChatFormat = dialog.querySelector('#includeChatFormat').checked;
        const includeVisionTokens = dialog.querySelector('#includeVisionTokens').checked;
        
        // Generate spatial tokens for each region
        const spatialTokens = this.regions.map(region => {
            if (region.type === 'bounding_box') {
                return `<|object_ref_start|>${region.label}<|object_ref_end|> at <|box_start|>${region.coords.join(',')}<|box_end|>`;
            } else if (region.type === 'polygon') {
                const coordString = region.coords.map(p => p.join(',')).join(' ');
                return `<|object_ref_start|>${region.label}<|object_ref_end|> outlined by <|quad_start|>${coordString}<|quad_end|>`;
            } else if (region.type === 'object_reference') {
                return `<|object_ref_start|>${region.label}<|object_ref_end|>`;
            }
        }).join(' and ');
        
        // Generate full prompt
        let fullPrompt = `${instruction} ${spatialTokens}`;
        
        if (includeVisionTokens) {
            fullPrompt += ` in this image: <|vision_start|><|image_pad|><|vision_end|>`;
        }
        
        if (includeChatFormat) {
            fullPrompt = `<|im_start|>user\n${fullPrompt}<|im_end|>`;
        }
        
        // Update outputs
        dialog.querySelector('#spatialTokensOutput').value = spatialTokens;
        dialog.querySelector('#fullPromptOutput').value = fullPrompt;
        
        this.updateDebugPanel(`Generated tokens for ${this.regions.length} regions:\n${spatialTokens}`);
    }
    
    getSelectedRegionIndex() {
        const selected = document.querySelector('#regionsList [data-region-index][style*="60"]');
        return selected ? parseInt(selected.dataset.regionIndex) : -1;
    }
    
    sendToNode(node, dialog) {
        if (!node) {
            this.updateDebugPanel("No connected node to send data to");
            return;
        }
        
        const fullPrompt = dialog.querySelector('#fullPromptOutput').value;
        const spatialTokens = dialog.querySelector('#spatialTokensOutput').value;
        
        // Find text input widget in the node and update it
        const textWidget = node.widgets?.find(w => 
            w.name === 'coordinates' || w.name === 'text' || w.name === 'input_text'
        );
        
        if (textWidget) {
            textWidget.value = dialog.querySelector('#instructionText').value;
            node.setDirtyCanvas(true, true);
            this.updateDebugPanel(`Sent prompt to node widget: ${textWidget.name}`);
        } else {
            this.updateDebugPanel("Could not find compatible text widget in node");
        }
    }
    
    exportJSON(dialog) {
        const exportData = {
            timestamp: new Date().toISOString(),
            image_dimensions: this.currentImage ? {
                width: this.currentImage.width,
                height: this.currentImage.height
            } : null,
            regions: this.regions,
            generated_tokens: {
                spatial_tokens: dialog.querySelector('#spatialTokensOutput').value,
                full_prompt: dialog.querySelector('#fullPromptOutput').value
            }
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `qwen_spatial_regions_${new Date().toISOString().slice(0, 19)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.updateDebugPanel("Exported regions and tokens as JSON");
    }
    
    exportAnnotatedImage(dialog) {
        if (!this.currentImage) {
            this.updateDebugPanel("No image to export");
            return;
        }
        
        // Create a new canvas for export
        const exportCanvas = document.createElement('canvas');
        exportCanvas.width = this.currentImage.width;
        exportCanvas.height = this.currentImage.height;
        const exportCtx = exportCanvas.getContext('2d');
        
        // Draw original image
        exportCtx.drawImage(this.currentImage, 0, 0);
        
        // Draw regions without scaling/offset
        const originalScale = this.imageScale;
        const originalOffset = { ...this.imageOffset };
        
        this.imageScale = 1;
        this.imageOffset = { x: 0, y: 0 };
        
        // Temporarily set context to export context
        const originalCtx = this.ctx;
        this.ctx = exportCtx;
        
        this.regions.forEach((region, index) => {
            this.drawRegion(region, index);
        });
        
        // Restore original values
        this.ctx = originalCtx;
        this.imageScale = originalScale;
        this.imageOffset = originalOffset;
        
        // Download the image
        exportCanvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `qwen_annotated_image_${new Date().toISOString().slice(0, 19)}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
        
        this.updateDebugPanel("Exported annotated image");
    }
    
    resizeCanvases(width, height) {
        if (this.canvas) {
            this.canvas.width = width;
            this.canvas.height = height;
        }
        if (this.imageCanvas) {
            this.imageCanvas.width = width;
            this.imageCanvas.height = height;
        }
    }
    
    updateDebugPanel(message) {
        if (!this.debugMode) return;
        
        const debugContent = document.querySelector('#debugContent');
        if (debugContent) {
            const timestamp = new Date().toLocaleTimeString();
            debugContent.textContent += `${timestamp}: ${message}\n`;
            debugContent.scrollTop = debugContent.scrollHeight;
        }
    }
}

// Register extension for the spatial editor node
app.registerExtension({
    name: "Comfy.QwenSpatialEditorEnhanced",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "QwenSpatialEditor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const node = this;
                
                // Add spatial editor button
                setTimeout(() => {
                    const editorBtn = node.addWidget("button", "Open Spatial Editor", "spatial", () => {
                        const editor = new QwenInteractiveSpatialEditor();
                        editor.createSpatialEditorDialog(node);
                    });
                    
                }, 10);
                
                return ret;
            };
        }
    }
});