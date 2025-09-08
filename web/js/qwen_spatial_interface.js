/**
 * Qwen Spatial Interface
 * Clean spatial token generation interface
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

class QwenSpatialInterface {
  constructor() {
    this.canvas = null;
    this.ctx = null;
    this.imageCanvas = null;
    this.imageCtx = null;
    this.regions = [];
    this.currentImage = null;
    this.isDrawing = false;
    this.drawingMode = "bounding_box";
    this.polygonPoints = [];
    this.imageScale = 1;
    this.imageOffset = { x: 0, y: 0 };
    this.node = null; // Store reference to the Python node
    
    // Qwen resolutions for coordinate optimization
    this.QWEN_RESOLUTIONS = [
      [1024, 1024],
      [672, 1568], [688, 1504], [720, 1456], [752, 1392],
      [800, 1328], [832, 1248], [880, 1184], [944, 1104],
      [1104, 944], [1184, 880], [1248, 832], [1328, 800],
      [1392, 752], [1456, 720], [1504, 688], [1568, 672],
      [1328, 1328], [1920, 1080], [1080, 1920],
    ];
    
    // Store original and optimized dimensions
    this.originalDimensions = { width: 0, height: 0 };
    this.optimizedDimensions = { width: 0, height: 0 };
  }

  createInterface(node = null) {
    console.log("=== CREATE INTERFACE START ===");
    console.log(`Node provided: ${!!node}`);
    this.node = node; // Store node reference for bidirectional sync
    
    const dialog = $el("div", {
      parent: document.body,
      style: {
        position: "fixed",
        top: "5%",
        left: "5%",
        width: "90%",
        height: "90%",
        background: "rgba(20, 20, 20, 0.95)",
        border: "1px solid rgba(255, 255, 255, 0.2)",
        borderRadius: "8px",
        zIndex: 10000,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        fontFamily: "system-ui, -apple-system, sans-serif",
        color: "rgba(255, 255, 255, 0.9)",
        backdropFilter: "blur(10px)",
      },
    });

    dialog.innerHTML = `
            <!-- Header -->
            <div style="
                background: rgba(0, 0, 0, 0.3);
                padding: 12px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            ">
                <h2 style="margin: 0; font-size: 18px; font-weight: 500;">Spatial Token Generator</h2>
                <button id="closeInterface" style="
                    padding: 6px 12px;
                    background: rgba(255, 255, 255, 0.1);
                    color: white;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                ">Close</button>
            </div>

            <!-- Main Content -->
            <div style="flex: 1; display: flex; overflow: hidden;">
                <!-- Left Panel -->
                <div style="
                    width: 280px;
                    background: rgba(0, 0, 0, 0.2);
                    border-right: 1px solid rgba(255, 255, 255, 0.1);
                    overflow-y: auto;
                    padding: 16px;
                ">
                    <!-- Image Load -->
                    <div style="margin-bottom: 20px;">
                        <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 500;">Load Image</h4>
                        <input type="file" id="imageUpload" accept="image/*" style="
                            width: 100%;
                            padding: 6px;
                            background: rgba(255, 255, 255, 0.05);
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            border-radius: 4px;
                            color: white;
                            font-size: 12px;
                        ">
                    </div>

                    <!-- Resolution Selection -->
                    <div id="resolutionSection" style="margin-bottom: 20px; display: none;">
                        <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 500;">Resolution</h4>
                        
                        <!-- Resolution Type -->
                        <div style="margin-bottom: 8px;">
                            <label style="display: block; margin-bottom: 4px; font-size: 12px; opacity: 0.8;">Selection Method:</label>
                            <select id="resolutionType" style="
                                width: 100%;
                                padding: 6px;
                                background: rgba(255, 255, 255, 0.05);
                                border: 1px solid rgba(255, 255, 255, 0.2);
                                border-radius: 4px;
                                color: white;
                                font-size: 12px;
                            ">
                                <option value="auto">Auto (Best Match)</option>
                                <option value="recommended">Recommended List</option>
                                <option value="custom">Custom Dimensions</option>
                            </select>
                        </div>

                        <!-- Recommended Resolutions -->
                        <div id="recommendedResolutions" style="margin-bottom: 8px; display: none;">
                            <label style="display: block; margin-bottom: 4px; font-size: 12px; opacity: 0.8;">Recommended:</label>
                            <select id="resolutionSelect" style="
                                width: 100%;
                                padding: 6px;
                                background: rgba(255, 255, 255, 0.05);
                                border: 1px solid rgba(255, 255, 255, 0.2);
                                border-radius: 4px;
                                color: white;
                                font-size: 12px;
                            ">
                            </select>
                        </div>

                        <!-- Custom Resolution -->
                        <div id="customResolution" style="display: none;">
                            <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                                <input type="number" id="customWidth" placeholder="Width" min="512" max="2048" step="16" style="
                                    flex: 1;
                                    padding: 6px;
                                    background: rgba(255, 255, 255, 0.05);
                                    border: 1px solid rgba(255, 255, 255, 0.2);
                                    border-radius: 4px;
                                    color: white;
                                    font-size: 12px;
                                ">
                                <input type="number" id="customHeight" placeholder="Height" min="512" max="2048" step="16" style="
                                    flex: 1;
                                    padding: 6px;
                                    background: rgba(255, 255, 255, 0.05);
                                    border: 1px solid rgba(255, 255, 255, 0.2);
                                    border-radius: 4px;
                                    color: white;
                                    font-size: 12px;
                                ">
                            </div>
                            <button id="applyCustomResolution" style="
                                width: 100%;
                                padding: 6px;
                                background: rgba(70, 130, 255, 0.3);
                                color: white;
                                border: 1px solid rgba(70, 130, 255, 0.5);
                                border-radius: 4px;
                                cursor: pointer;
                                font-size: 12px;
                            ">Apply Custom</button>
                        </div>

                        <!-- Resize Method -->
                        <div id="resizeMethodSection" style="margin-bottom: 8px;">
                            <label style="display: block; margin-bottom: 4px; font-size: 12px; opacity: 0.8;">Resize Method:</label>
                            <select id="resizeMethod" style="
                                width: 100%;
                                padding: 6px;
                                background: rgba(255, 255, 255, 0.05);
                                border: 1px solid rgba(255, 255, 255, 0.2);
                                border-radius: 4px;
                                color: white;
                                font-size: 12px;
                            ">
                                <option value="pad">Pad (scale + black borders)</option>
                                <option value="crop">Crop (scale + center crop)</option>
                                <option value="stretch">Stretch (distort to exact size)</option>
                                <option value="resize">Resize (scale, maintain ratio)</option>
                            </select>
                            <div style="
                                margin-top: 4px;
                                font-size: 10px;
                                opacity: 0.6;
                                line-height: 1.3;
                            " id="resizeMethodHelp">
                                Scale and add black padding to fit exact dimensions
                            </div>
                        </div>

                        <!-- Resolution Info -->
                        <div id="resolutionInfo" style="
                            margin-top: 8px;
                            padding: 6px;
                            background: rgba(255, 255, 255, 0.05);
                            border-radius: 4px;
                            font-size: 11px;
                            opacity: 0.8;
                        ">
                            <div>Original: <span id="originalSize">-</span></div>
                            <div>Target: <span id="targetSize">-</span></div>
                            <div>Aspect Ratio: <span id="aspectRatioInfo">-</span></div>
                        </div>
                    </div>

                    <!-- Drawing Mode -->
                    <div style="margin-bottom: 20px;">
                        <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 500;">Spatial Type</h4>
                        <select id="spatialType" style="
                            width: 100%;
                            padding: 6px;
                            background: rgba(255, 255, 255, 0.05);
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            border-radius: 4px;
                            color: white;
                            font-size: 12px;
                        ">
                            <option value="bounding_box">Bounding Box</option>
                            <option value="polygon">Polygon</option>
                            <option value="object_reference">Object Reference</option>
                        </select>

                        <div style="margin-top: 8px;">
                            <label style="display: block; margin-bottom: 4px; font-size: 12px; opacity: 0.8;">Label:</label>
                            <input type="text" id="regionLabel" value="object" style="
                                width: 100%;
                                padding: 6px;
                                background: rgba(255, 255, 255, 0.05);
                                border: 1px solid rgba(255, 255, 255, 0.2);
                                border-radius: 4px;
                                color: white;
                                font-size: 12px;
                            ">
                        </div>
                        <div style="margin-top: 8px;">
                            <label style="display: flex; align-items: center; font-size: 12px; opacity: 0.8; cursor: pointer;">
                                <input type="checkbox" id="includeObjectRef" checked style="
                                    margin-right: 6px;
                                    cursor: pointer;
                                ">
                                Include object reference label (for boxes/polygons)
                            </label>
                        </div>

                        <div id="drawingHelp" style="
                            margin-top: 8px;
                            padding: 8px;
                            background: rgba(255, 255, 255, 0.05);
                            border-radius: 4px;
                            font-size: 11px;
                            opacity: 0.7;
                        ">
                            Click and drag to create bounding boxes
                        </div>

                        <div style="margin-top: 8px;">
                            <label style="display: flex; align-items: center; font-size: 12px; cursor: pointer;">
                                <input type="checkbox" id="debugMode" style="margin-right: 6px;">
                                Debug Mode
                            </label>
                        </div>

                        <!-- Dynamic controls area -->
                        <div id="dynamicControls" style="margin-top: 8px;"></div>
                    </div>

                    <!-- Regions List -->
                    <div style="margin-bottom: 20px;">
                        <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 500;">Regions</h4>
                        <div id="regionsList" style="
                            max-height: 150px;
                            overflow-y: auto;
                            border: 1px solid rgba(255, 255, 255, 0.1);
                            border-radius: 4px;
                            padding: 6px;
                            background: rgba(255, 255, 255, 0.02);
                        ">
                            <div style="text-align: center; padding: 20px; opacity: 0.5; font-size: 12px;">
                                No regions created
                            </div>
                        </div>

                        <button id="clearRegions" style="
                            width: 100%;
                            padding: 6px;
                            margin-top: 8px;
                            background: rgba(255, 255, 255, 0.1);
                            color: white;
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 12px;
                        ">Clear All</button>
                    </div>

                    <!-- Generate -->
                    <div>
                        <button id="generateTokens" style="
                            width: 100%;
                            padding: 10px;
                            background: rgba(0, 120, 255, 0.8);
                            color: white;
                            border: 1px solid rgba(0, 120, 255, 0.5);
                            border-radius: 4px;
                            cursor: pointer;
                            font-weight: 500;
                        ">Generate Spatial Tokens</button>
                    </div>
                </div>

                <!-- Canvas -->
                <div style="
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    background: rgba(40, 40, 40, 0.3);
                ">
                    <!-- Canvas Controls -->
                    <div style="
                        background: rgba(0, 0, 0, 0.2);
                        padding: 8px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        gap: 12px;
                        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    ">
                        <button id="zoomIn" style="
                            padding: 4px 8px;
                            background: rgba(255, 255, 255, 0.1);
                            color: white;
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            border-radius: 3px;
                            cursor: pointer;
                            font-size: 11px;
                        ">Zoom +</button>
                        <span id="zoomLevel" style="font-size: 11px; opacity: 0.7;">100%</span>
                        <button id="zoomOut" style="
                            padding: 4px 8px;
                            background: rgba(255, 255, 255, 0.1);
                            color: white;
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            border-radius: 3px;
                            cursor: pointer;
                            font-size: 11px;
                        ">Zoom -</button>
                        <div style="margin-left: 15px;">
                            <span id="mouseCoords" style="font-size: 11px; opacity: 0.6;">Mouse: (0, 0)</span>
                        </div>
                    </div>

                    <!-- Drawing Area -->
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

                <!-- Right Panel -->
                <div style="
                    width: 320px;
                    background: rgba(0, 0, 0, 0.2);
                    border-left: 1px solid rgba(255, 255, 255, 0.1);
                    overflow-y: auto;
                    padding: 16px;
                ">
                    <h4 style="margin: 0 0 12px 0; font-size: 14px; font-weight: 500;">Generated Tokens</h4>

                    <textarea id="spatialTokensOutput" rows="8" readonly style="
                        width: 100%;
                        padding: 8px;
                        background: rgba(255, 255, 255, 0.05);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        border-radius: 4px;
                        color: white;
                        font-family: 'Monaco', 'Menlo', monospace;
                        font-size: 11px;
                        resize: vertical;
                        margin-bottom: 8px;
                    " placeholder="Generated spatial tokens will appear here..."></textarea>

                    <div style="display: flex; gap: 6px; margin-bottom: 16px;">
                        <button id="copyTokens" style="
                            flex: 1;
                            padding: 6px;
                            background: rgba(255, 255, 255, 0.1);
                            color: white;
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 12px;
                        ">Copy</button>
                        <button id="sendToNode" style="
                            flex: 1;
                            padding: 6px;
                            background: rgba(0, 120, 255, 0.8);
                            color: white;
                            border: 1px solid rgba(0, 120, 255, 0.5);
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 12px;
                        ">Send to Node</button>
                    </div>

                    <div id="debugOutput" style="
                        background: rgba(0, 0, 0, 0.3);
                        padding: 8px;
                        border-radius: 4px;
                        font-family: 'Monaco', 'Menlo', monospace;
                        font-size: 10px;
                        max-height: 200px;
                        overflow-y: auto;
                        opacity: 0.8;
                        white-space: pre-wrap;
                        display: none;
                    ">Ready to load image...</div>
                </div>
            </div>
        `;

    this.setupEventListeners(dialog, node);
    this.setupCanvas(dialog);
    
    // Try to auto-load connected image
    setTimeout(() => {
      this.loadImageFromConnectedNode(dialog);
    }, 500);
    
    return dialog;
  }

  setupEventListeners(dialog, node) {
    // Close
    dialog.querySelector("#closeInterface").onclick = () => {
      document.body.removeChild(dialog);
    };

    // Image upload
    dialog.querySelector("#imageUpload").onchange = (e) => {
      const file = e.target.files[0];
      if (file) this.loadImage(file, dialog);
    };

    // Spatial type change
    dialog.querySelector("#spatialType").onchange = (e) => {
      this.drawingMode = e.target.value;
      this.polygonPoints = [];
      this.updateDrawingHelp(dialog);
      this.updateDynamicControls(dialog);
      this.redrawCanvas();
    };

    // Debug mode toggle
    dialog.querySelector("#debugMode").onchange = (e) => {
      const debugOutput = dialog.querySelector("#debugOutput");
      debugOutput.style.display = e.target.checked ? "block" : "none";
    };

    // Resolution type selection
    dialog.querySelector("#resolutionType").onchange = (e) => {
      this.updateResolutionInterface(e.target.value, dialog);
    };

    // Recommended resolution selection  
    dialog.querySelector("#resolutionSelect").onchange = (e) => {
      const [width, height] = e.target.value.split('x').map(Number);
      this.applyResolution(width, height, dialog);
    };

    // Resize method change
    dialog.querySelector("#resizeMethod").onchange = (e) => {
      this.updateResizeMethodHelp(e.target.value, dialog);
      
      // If an image is loaded, re-apply the optimization with the new method
      if (this.currentImage && this.originalImageSrc) {
        this.updateDebug(`Resize method changed to: ${e.target.value}`, dialog);
        const currentWidth = this.optimizedDimensions.width;
        const currentHeight = this.optimizedDimensions.height;
        this.applyResolution(currentWidth, currentHeight, dialog);
      }
    };

    // Custom resolution application
    dialog.querySelector("#applyCustomResolution").onclick = () => {
      this.updateDebug("=== CUSTOM RESOLUTION BUTTON CLICKED ===", dialog);
      
      const widthInput = dialog.querySelector("#customWidth");
      const heightInput = dialog.querySelector("#customHeight");
      
      this.updateDebug(`Width input element: ${!!widthInput}, value: '${widthInput?.value}'`, dialog);
      this.updateDebug(`Height input element: ${!!heightInput}, value: '${heightInput?.value}'`, dialog);
      
      const width = parseInt(widthInput?.value);
      const height = parseInt(heightInput?.value);
      
      this.updateDebug(`Parsed values: width=${width}, height=${height}`, dialog);
      this.updateDebug(`Width isNaN: ${isNaN(width)}, Height isNaN: ${isNaN(height)}`, dialog);
      
      if (isNaN(width) || isNaN(height)) {
        this.updateDebug("ERROR: Please enter valid numeric values for both width and height", dialog);
        return;
      }
      
      if (width < 512 || height < 512) {
        this.updateDebug("ERROR: Minimum resolution is 512x512", dialog);
        return;
      }
      
      // Ensure dimensions are multiples of 16
      const adjustedWidth = Math.round(width / 16) * 16;
      const adjustedHeight = Math.round(height / 16) * 16;
      
      if (adjustedWidth !== width || adjustedHeight !== height) {
        this.updateDebug(`Adjusted resolution to multiples of 16: ${width}x${height} → ${adjustedWidth}x${adjustedHeight}`, dialog);
      }
      
      this.updateDebug(`About to call applyResolution(${adjustedWidth}, ${adjustedHeight})`, dialog);
      this.applyResolution(adjustedWidth, adjustedHeight, dialog);
    };

    // Clear regions
    dialog.querySelector("#clearRegions").onclick = () => {
      this.regions = [];
      this.updateRegionsList(dialog);
      this.redrawCanvas();
      this.generateTokens(dialog); // Auto-generate tokens (will be empty)
    };

    // Generate tokens
    dialog.querySelector("#generateTokens").onclick = () => {
      this.generateTokens(dialog);
    };

    // Copy tokens
    dialog.querySelector("#copyTokens").onclick = () => {
      const tokens = dialog.querySelector("#spatialTokensOutput").value;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(tokens);
        this.updateDebug("Tokens copied to clipboard", dialog);
      } else {
        // Fallback for older browsers
        const textArea = document.createElement("textarea");
        textArea.value = tokens;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand("copy");
        document.body.removeChild(textArea);
        this.updateDebug("Tokens copied to clipboard (fallback)", dialog);
      }
    };

    // Send to node
    dialog.querySelector("#sendToNode").onclick = () => {
      this.sendToNode(node, dialog);
    };

    // Zoom controls
    dialog.querySelector("#zoomIn").onclick = () => {
      this.imageScale = Math.min(this.imageScale * 1.2, 5);
      this.updateCanvas(dialog);
    };

    dialog.querySelector("#zoomOut").onclick = () => {
      this.imageScale = Math.max(this.imageScale / 1.2, 0.1);
      this.updateCanvas(dialog);
    };
  }

  setupCanvas(dialog) {
    const container = dialog.querySelector("#canvasContainer");
    this.canvas = dialog.querySelector("#drawingCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.imageCanvas = dialog.querySelector("#imageCanvas");
    this.imageCtx = this.imageCanvas.getContext("2d");

    this.resizeCanvases(container.clientWidth, container.clientHeight);
    this.setupCanvasEvents(dialog);

    new ResizeObserver(() => {
      this.resizeCanvases(container.clientWidth, container.clientHeight);
      this.updateCanvas(dialog);
    }).observe(container);
  }

  setupCanvasEvents(dialog) {
    let startX,
      startY,
      isDragging = false;

    this.canvas.addEventListener("mousedown", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      startX = (e.clientX - rect.left) / this.imageScale;
      startY = (e.clientY - rect.top) / this.imageScale;

      if (this.drawingMode === "polygon") {
        this.polygonPoints.push([Math.round(startX), Math.round(startY)]);
        this.redrawCanvas();
        this.updateDynamicControls(dialog);
        if (e.detail === 2 && this.polygonPoints.length >= 3) {
          this.finishPolygon(dialog);
        }
      } else if (this.drawingMode === "object_reference") {
        this.addObjectReference(startX, startY, dialog);
      } else {
        isDragging = true;
      }
    });

    this.canvas.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const currentX = (e.clientX - rect.left) / this.imageScale;
      const currentY = (e.clientY - rect.top) / this.imageScale;

      dialog.querySelector("#mouseCoords").textContent =
        `Mouse: (${Math.round(currentX)}, ${Math.round(currentY)})`;

      if (isDragging && this.drawingMode === "bounding_box") {
        this.redrawCanvas();
        this.drawPreviewBox(startX, startY, currentX, currentY);
      }
    });

    this.canvas.addEventListener("mouseup", (e) => {
      if (!isDragging) return;

      const rect = this.canvas.getBoundingClientRect();
      const endX = (e.clientX - rect.left) / this.imageScale;
      const endY = (e.clientY - rect.top) / this.imageScale;

      if (this.drawingMode === "bounding_box") {
        if (Math.abs(endX - startX) > 5 && Math.abs(endY - startY) > 5) {
          this.addBoundingBox(startX, startY, endX, endY, dialog);
        }
      }

      isDragging = false;
    });
  }

  loadImage(file, dialog) {
    console.log("=== SPATIAL INTERFACE: LOAD IMAGE START ===");
    console.log(`Loading file: ${file.name}, size: ${file.size} bytes, type: ${file.type}`);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      console.log(`FileReader loaded: ${e.target.result.length} characters`);
      // Store original image source for re-optimization
      this.originalImageSrc = e.target.result;
      console.log("Stored original image source for re-optimization");
      
      const img = new Image();
      img.onload = () => {
        console.log(`Original image loaded: ${img.width}x${img.height}`);
        // Show resolution section now that image is loaded
        dialog.querySelector('#resolutionSection').style.display = 'block';
        
        // Get current resize method (default to 'pad')
        const resizeMethodSelect = dialog.querySelector("#resizeMethod");
        const resizeMethod = resizeMethodSelect ? resizeMethodSelect.value : 'pad';
        console.log(`Using resize method: ${resizeMethod}`);
        
        // Optimize the image for Qwen coordinate system
        console.log("Starting image optimization for Qwen...");
        const optimizedData = this.optimizeImageForQwen(img, null, null, resizeMethod);
        
        // Create new image from optimized canvas
        this.currentImage = new Image();
        this.currentImage.onload = () => {
          console.log(`Optimized image loaded: ${this.currentImage.width}x${this.currentImage.height}`);
          console.log(`Optimization data - scale: ${optimizedData.scale}, offset: (${optimizedData.offsetX},${optimizedData.offsetY})`);
          console.log(`Final dimensions - original: ${this.originalDimensions.width}x${this.originalDimensions.height}, optimized: ${this.optimizedDimensions.width}x${this.optimizedDimensions.height}`);
          
          this.imageScale = 1;
          this.regions = [];
          console.log("Calling updateCanvas, updateRegionsList, updateResolutionInfo...");
          this.updateCanvas(dialog);
          this.updateRegionsList(dialog);
          this.updateResolutionInfo(dialog);
          this.updateDebug(`Original: ${this.originalDimensions.width}x${this.originalDimensions.height}px`, dialog);
          this.updateDebug(`Optimized: ${this.optimizedDimensions.width}x${this.optimizedDimensions.height}px`, dialog);
          this.updateDebug(`Scale: ${optimizedData.scale.toFixed(3)}, Offset: (${optimizedData.offsetX.toFixed(0)},${optimizedData.offsetY.toFixed(0)})`, dialog);
          this.updateDebug(`Image loaded and optimized - draw regions to generate tokens`, dialog);
          
          console.log("=== SPATIAL INTERFACE: LOAD IMAGE COMPLETE ===");
        };
        console.log("Setting optimized canvas as image source...");
        this.currentImage.src = optimizedData.canvas.toDataURL();
        
        // Store optimization data for coordinate calculations
        this.optimizationData = optimizedData;
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  updateCanvas(dialog) {
    if (!this.currentImage) return;

    this.imageCtx.clearRect(
      0,
      0,
      this.imageCanvas.width,
      this.imageCanvas.height,
    );

    const imgWidth = this.currentImage.width * this.imageScale;
    const imgHeight = this.currentImage.height * this.imageScale;
    this.imageCtx.drawImage(this.currentImage, 0, 0, imgWidth, imgHeight);

    dialog.querySelector("#zoomLevel").textContent =
      `${Math.round(this.imageScale * 100)}%`;
    this.redrawCanvas();
  }

  redrawCanvas() {
    if (!this.canvas) return;

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    this.regions.forEach((region, index) => {
      const color = ["#ff4444", "#44ff44", "#4444ff", "#ffff44"][index % 4];
      this.drawRegion(region, color);
    });

    if (this.drawingMode === "polygon" && this.polygonPoints.length > 0) {
      this.drawPolygonInProgress();
    }
  }

  drawRegion(region, color) {
    this.ctx.strokeStyle = color;
    this.ctx.fillStyle = color + "20";
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([]);

    if (region.type === "bounding_box") {
      const [x1, y1, x2, y2] = region.coords;
      const drawX1 = x1 * this.imageScale;
      const drawY1 = y1 * this.imageScale;
      const drawWidth = (x2 - x1) * this.imageScale;
      const drawHeight = (y2 - y1) * this.imageScale;

      this.ctx.strokeRect(drawX1, drawY1, drawWidth, drawHeight);
      this.ctx.fillRect(drawX1, drawY1, drawWidth, drawHeight);

      this.ctx.fillStyle = color;
      this.ctx.font = "12px system-ui";
      this.ctx.fillText(region.label, drawX1, drawY1 - 5);
    } else if (region.type === "object_reference") {
      const [x, y] = region.coords;
      const drawX = x * this.imageScale;
      const drawY = y * this.imageScale;

      // Draw a colored circle for object reference
      this.ctx.beginPath();
      this.ctx.arc(drawX, drawY, 8, 0, 2 * Math.PI);
      this.ctx.fillStyle = color;
      this.ctx.fill();
      this.ctx.stroke();

      // Add a white center dot
      this.ctx.beginPath();
      this.ctx.arc(drawX, drawY, 3, 0, 2 * Math.PI);
      this.ctx.fillStyle = "white";
      this.ctx.fill();

      // Label
      this.ctx.fillStyle = color;
      this.ctx.font = "12px system-ui";
      this.ctx.fillText(region.label, drawX + 12, drawY - 5);
    } else if (region.type === "polygon") {
      if (region.coords.length >= 3) {
        this.ctx.beginPath();
        const [startX, startY] = region.coords[0];
        this.ctx.moveTo(startX * this.imageScale, startY * this.imageScale);

        for (let i = 1; i < region.coords.length; i++) {
          const [x, y] = region.coords[i];
          this.ctx.lineTo(x * this.imageScale, y * this.imageScale);
        }

        this.ctx.closePath();
        this.ctx.stroke();
        this.ctx.fill();

        // Draw snap points
        region.coords.forEach(([x, y]) => {
          const drawX = x * this.imageScale;
          const drawY = y * this.imageScale;

          this.ctx.beginPath();
          this.ctx.arc(drawX, drawY, 4, 0, 2 * Math.PI);
          this.ctx.fillStyle = color;
          this.ctx.fill();
        });

        // Label at first point
        const [labelX, labelY] = region.coords[0];
        this.ctx.fillStyle = color;
        this.ctx.font = "12px system-ui";
        this.ctx.fillText(
          region.label,
          labelX * this.imageScale,
          labelY * this.imageScale - 15,
        );
      }
    }
  }

  drawPreviewBox(x1, y1, x2, y2) {
    this.ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
    this.ctx.setLineDash([5, 5]);
    this.ctx.lineWidth = 1;

    const drawX1 = x1 * this.imageScale;
    const drawY1 = y1 * this.imageScale;
    const drawWidth = (x2 - x1) * this.imageScale;
    const drawHeight = (y2 - y1) * this.imageScale;

    this.ctx.strokeRect(drawX1, drawY1, drawWidth, drawHeight);
    this.ctx.setLineDash([]);
  }

  addBoundingBox(x1, y1, x2, y2, dialog) {
    const label = dialog.querySelector("#regionLabel").value || "object";
    const includeObjectRef = dialog.querySelector("#includeObjectRef").checked;

    const region = {
      type: "bounding_box",
      label: label,
      includeObjectRef: includeObjectRef,
      coords: [
        Math.min(x1, x2),
        Math.min(y1, y2),
        Math.max(x1, x2),
        Math.max(y1, y2),
      ],
    };

    this.regions.push(region);
    this.updateRegionsList(dialog);
    this.redrawCanvas();
    this.generateTokens(dialog); // Auto-generate tokens
    this.updateDebug(`Added bounding box: ${label} (obj_ref: ${includeObjectRef})`, dialog);
  }

  updateRegionsList(dialog) {
    const listContainer = dialog.querySelector("#regionsList");

    if (this.regions.length === 0) {
      listContainer.innerHTML =
        '<div style="text-align: center; padding: 20px; opacity: 0.5; font-size: 12px;">No regions created</div>';
      return;
    }

    const listHTML = this.regions
      .map((region, index) => {
        const color = ["#ff4444", "#44ff44", "#4444ff", "#ffff44"][index % 4];
        return `
                <div style="
                    padding: 6px;
                    margin-bottom: 4px;
                    border: 1px solid ${color}40;
                    border-radius: 3px;
                    background: ${color}10;
                    font-size: 12px;
                    position: relative;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <input type="text" 
                                   value="${region.label}" 
                                   data-region-index="${index}"
                                   class="region-label-input"
                                   style="
                                       background: transparent;
                                       border: none;
                                       color: ${color};
                                       font-weight: 500;
                                       font-size: 12px;
                                       width: 100%;
                                       padding: 1px 0;
                                       outline: none;
                                   "
                                   onchange="this.dispatchEvent(new CustomEvent('regionLabelChange', {detail: {index: ${index}, label: this.value}}))"
                                   onkeypress="if(event.key==='Enter') this.blur()">
                            <div style="opacity: 0.7; font-size: 10px;">${region.type}</div>
                        </div>
                        <div style="display: flex; gap: 4px; margin-left: 8px;">
                            <button class="region-delete-btn" 
                                    data-region-index="${index}"
                                    style="
                                        padding: 2px 6px;
                                        background: rgba(255, 0, 0, 0.2);
                                        border: 1px solid rgba(255, 0, 0, 0.3);
                                        border-radius: 2px;
                                        color: #ff6666;
                                        font-size: 10px;
                                        cursor: pointer;
                                        line-height: 1;
                                    "
                                    title="Delete region">×</button>
                        </div>
                    </div>
                </div>
            `;
      })
      .join("");

    listContainer.innerHTML = listHTML;
    
    // Add event listeners for the new functionality
    this.setupRegionListEvents(dialog);
  }

  setupRegionListEvents(dialog) {
    // Handle delete button clicks
    dialog.querySelectorAll('.region-delete-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const index = parseInt(e.target.dataset.regionIndex);
        this.deleteRegion(index, dialog);
      });
    });

    // Handle label changes
    dialog.querySelectorAll('.region-label-input').forEach(input => {
      input.addEventListener('regionLabelChange', (e) => {
        const { index, label } = e.detail;
        this.updateRegionLabel(index, label, dialog);
      });
      
      // Also handle blur to catch changes without Enter key
      input.addEventListener('blur', (e) => {
        const index = parseInt(e.target.dataset.regionIndex);
        const label = e.target.value;
        this.updateRegionLabel(index, label, dialog);
      });
    });
  }

  deleteRegion(index, dialog) {
    if (index < 0 || index >= this.regions.length) return;
    
    const deletedRegion = this.regions[index];
    
    // Simple confirmation
    if (!confirm(`Delete region "${deletedRegion.label}"?`)) return;
    
    this.regions.splice(index, 1);
    
    this.updateRegionsList(dialog);
    this.redrawCanvas();
    this.generateTokens(dialog);
    
    this.updateDebug(`Deleted region: ${deletedRegion.label} (${deletedRegion.type})`, dialog);
  }

  updateRegionLabel(index, newLabel, dialog) {
    if (index < 0 || index >= this.regions.length) return;
    
    // Prevent empty labels - revert to old label
    if (!newLabel || newLabel.trim() === '') {
      const input = dialog.querySelector(`input[data-region-index="${index}"]`);
      if (input) input.value = this.regions[index].label;
      return;
    }
    
    const oldLabel = this.regions[index].label;
    const trimmedLabel = newLabel.trim();
    
    // Only update if label actually changed
    if (oldLabel !== trimmedLabel) {
      this.regions[index].label = trimmedLabel;
      this.redrawCanvas();
      this.generateTokens(dialog);
      this.updateDebug(`Updated region label: "${oldLabel}" → "${trimmedLabel}"`, dialog);
    }
  }

  generateTokens(dialog) {
    console.log("=== GENERATE TOKENS START ===");
    console.log(`generateTokens called: image=${!!this.currentImage}, regions=${this.regions.length}`);
    
    this.updateDebug(`generateTokens called: image=${!!this.currentImage}, regions=${this.regions.length}`, dialog);
    
    if (!this.currentImage) {
      console.log("No image loaded - cannot generate tokens");
      dialog.querySelector('#spatialTokensOutput').value = '';
      this.updateDebug("No image loaded - cannot generate tokens", dialog);
      return;
    }
    
    if (this.regions.length === 0) {
      console.log("No regions to generate tokens from");
      dialog.querySelector('#spatialTokensOutput').value = '';
      this.updateDebug("No regions to generate tokens from", dialog);
      return;
    }
    
    // Use optimized dimensions for coordinate calculation
    const imageWidth = this.optimizedDimensions.width || this.currentImage.width;
    const imageHeight = this.optimizedDimensions.height || this.currentImage.height;
    
    // Calculate native resolution dimensions (multiples of 28 for ViT patches)
    const nativeWidth = Math.ceil(imageWidth / 28) * 28;
    const nativeHeight = Math.ceil(imageHeight / 28) * 28;
    
    console.log(`Using dimensions for coordinate calculation: ${imageWidth}x${imageHeight}`);
    console.log(`Native ViT dimensions (multiples of 28): ${nativeWidth}x${nativeHeight}`);
    console.log(`Current image dimensions: ${this.currentImage.width}x${this.currentImage.height}`);
    console.log(`Optimized dimensions: ${this.optimizedDimensions.width}x${this.optimizedDimensions.height}`);
    
    this.updateDebug(`Using dimensions for normalization: ${imageWidth}x${imageHeight}`, dialog);
    
    console.log(`Processing ${this.regions.length} regions for token generation...`);
    const tokens = this.regions.map((region, index) => {
      console.log(`Processing region ${index + 1}: ${region.type} "${region.label}"`);
      
      if (region.type === 'bounding_box') {
        const [x1, y1, x2, y2] = region.coords;
        
        // COMPREHENSIVE COORDINATE DEBUGGING (FIXED: Using absolute pixels with native scaling)
        console.log(`=== COORDINATE DEBUG: ${region.label} ===`);
        console.log(`Raw canvas coordinates: (${x1},${y1},${x2},${y2})`);
        console.log(`Canvas dimensions: ${imageWidth}x${imageHeight}`);
        console.log(`Native ViT dimensions: ${nativeWidth}x${nativeHeight}`);
        
        // Scale coordinates to native ViT resolution (what model actually sees)
        const scaleX = nativeWidth / imageWidth;
        const scaleY = nativeHeight / imageHeight;
        
        const nativeX1 = Math.round(x1 * scaleX);
        const nativeY1 = Math.round(y1 * scaleY);
        const nativeX2 = Math.round(x2 * scaleX);
        const nativeY2 = Math.round(y2 * scaleY);
        
        console.log(`Scale factors: X=${scaleX.toFixed(3)}, Y=${scaleY.toFixed(3)}`);
        console.log(`Native coordinates: (${nativeX1},${nativeY1},${nativeX2},${nativeY2})`);
        console.log(`Box size: ${nativeX2-nativeX1}x${nativeY2-nativeY1} native pixels`);
        console.log(`=== END COORDINATE DEBUG ===`);
        
        this.updateDebug(`Box ${region.label}: native(${nativeX1},${nativeY1},${nativeX2},${nativeY2})`, dialog);
        
        // Use Qwen-Image Edit format with native ViT coordinates
        let token;
        if (region.includeObjectRef !== false) {
          token = `<|object_ref_start|>${region.label}<|object_ref_end|> at <|box_start|>${nativeX1},${nativeY1},${nativeX2},${nativeY2}<|box_end|>`;
          console.log(`Generated bounding box token with object ref: ${token}`);
        } else {
          token = `<|box_start|>${nativeX1},${nativeY1},${nativeX2},${nativeY2}<|box_end|>`;
          console.log(`Generated bounding box token without object ref: ${token}`);
        }
        return token;
        
      } else if (region.type === 'object_reference') {
        console.log(`Object reference - label only: ${region.label}`);
        // Object reference is just a label with no coordinates
        const token = `<|object_ref_start|>${region.label}<|object_ref_end|>`;
        console.log(`Generated object reference token: ${token}`);
        return token;
        
      } else if (region.type === 'polygon') {
        console.log(`Polygon with ${region.coords.length} points: ${JSON.stringify(region.coords)}`);
        
        // Convert to native ViT coordinates for quad format
        // Format: (x1,y1),(x2,y2),(x3,y3),(x4,y4) like the Chinese example
        const nativePoints = region.coords.map(([x, y]) => {
          const nativeX = Math.round(x * (nativeWidth / imageWidth));
          const nativeY = Math.round(y * (nativeHeight / imageHeight));
          return `(${nativeX},${nativeY})`;
        }).join(',');
        
        console.log(`Native ViT polygon coordinates: ${nativePoints}`);
        
        this.updateDebug(`Polygon ${region.label}: ${region.coords.length} points scaled to native`, dialog);
        
        // Use quad format with native coordinates - format: <|quad_start|>(x1,y1),(x2,y2),...<|quad_end|>
        let token;
        if (region.includeObjectRef !== false) {
          token = `<|object_ref_start|>${region.label}<|object_ref_end|><|quad_start|>${nativePoints}<|quad_end|>`;
          console.log(`Generated polygon token with object ref: ${token}`);
        } else {
          token = `<|quad_start|>${nativePoints}<|quad_end|>`;
          console.log(`Generated polygon token without object ref: ${token}`);
        }
        return token;
      }
      
      console.log(`Unknown region type: ${region.type}`);
      return ''; // fallback
    }).filter(token => token); // remove empty tokens
    
    console.log(`Generated ${tokens.length} tokens before joining`);
    console.log(`Individual tokens:`, tokens);
    const spatialTokens = tokens.join(' ');
    console.log(`Final spatial tokens: ${spatialTokens.length} characters`);
    console.log(`Final spatial tokens content: ${spatialTokens}`);
    
    dialog.querySelector('#spatialTokensOutput').value = spatialTokens;
    this.updateDebug(`Generated ${tokens.length} spatial tokens`, dialog);
    console.log("=== GENERATE TOKENS END ===");
    
    // Auto-sync spatial tokens to Python node
    const syncSuccess = this.syncSpatialTokensToNode();
    if (syncSuccess) {
      console.log("Spatial tokens auto-synced to Python node");
    } else {
      console.log("Failed to auto-sync spatial tokens");
    }
    
    // Update base_prompt widget with spatial tokens for easy editing  
    this.updateBasePromptWithTokens(spatialTokens);
  }

  updateBasePromptWithTokens(spatialTokens) {
    console.log("=== UPDATE BASE PROMPT WITH TOKENS START ===");
    console.log(`Node: ${!!this.node}, has widgets: ${!!(this.node && this.node.widgets)}`);
    
    if (!this.node || !this.node.widgets) {
      console.log("No node or widgets available for base prompt update");
      return;
    }
    
    const basePromptWidget = this.node.widgets.find(w => w.name === 'base_prompt');
    console.log(`Base prompt widget found: ${!!basePromptWidget}`);
    
    if (!basePromptWidget) {
      console.log("No base_prompt widget found");
      return;
    }
    
    // Get current base prompt, removing any existing spatial tokens
    let currentPrompt = basePromptWidget.value || '';
    console.log(`Current base prompt: ${currentPrompt.length} characters`);
    console.log(`Current base prompt content: ${currentPrompt}`);
    
    // Remove existing spatial tokens (anything with <|...|> patterns) and coordinate text
    let cleanPrompt = currentPrompt.replace(/<\|[^|]+\|>/g, '').trim();
    // Also remove coordinate text patterns like "object at 28,296,321,690"
    cleanPrompt = cleanPrompt.replace(/\b\w+\s+at\s+\d+,\d+,\d+,\d+\b/g, '').trim();
    // Remove extra whitespace
    cleanPrompt = cleanPrompt.replace(/\s+/g, ' ').trim();
    
    console.log(`Clean prompt after token removal: ${cleanPrompt.length} characters`);
    console.log(`Clean prompt content: ${cleanPrompt}`);
    
    // Combine base prompt with spatial tokens
    const combinedPrompt = spatialTokens ? 
      `${cleanPrompt} ${spatialTokens}`.trim() : 
      cleanPrompt;
      
    console.log(`Combined prompt: ${combinedPrompt.length} characters`);
    console.log(`Combined prompt content: ${combinedPrompt}`);
      
    basePromptWidget.value = combinedPrompt;
    this.node.setDirtyCanvas(true, true);
    console.log("Base prompt widget updated and node marked as dirty");
    console.log("=== UPDATE BASE PROMPT WITH TOKENS END ===");
  }

  loadImageFromConnectedNode(dialog) {
    console.log("=== LOAD IMAGE FROM CONNECTED NODE START ===");
    
    if (!this.node) {
      console.log("No node reference available");
      return;
    }

    // Look for connected image input
    const imageInput = this.node.inputs?.find(input => input.name === "image");
    
    if (!imageInput || !imageInput.link) {
      console.log("No image input connected to node");
      this.updateDebugMessage("No image connected - please connect an image to the node and upload it in this interface", dialog);
      return;
    }

    console.log(`Found connected image input: ${imageInput.name}`);
    
    // Get the connected node and output  
    const link = this.node.graph.links[imageInput.link];
    if (!link) {
      console.log("Link not found");
      return;
    }
    
    const sourceNode = this.node.graph.getNodeById(link.origin_id);
    
    if (!sourceNode) {
      console.log(`Source node ${link.origin_id} not found`);
      return;
    }

    console.log(`Found source node: ${sourceNode.title || sourceNode.type}`);
    
    // Show helpful message about the connected image
    const sourceInfo = `${sourceNode.title || sourceNode.type}`;
    this.updateDebugMessage(`Image connected from: ${sourceInfo}`, dialog);
    this.updateDebugMessage(`Please upload the same image in this interface for spatial editing`, dialog);
    this.updateDebugMessage(`The coordinates will be automatically matched to the connected image`, dialog);
    
    console.log("=== LOAD IMAGE FROM CONNECTED NODE END ===");
  }

  updateDebugMessage(message, dialog) {
    if (dialog && dialog.querySelector) {
      const debugOutput = dialog.querySelector("#debugOutput");
      if (debugOutput) {
        const timestamp = new Date().toLocaleTimeString();
        debugOutput.textContent += `${timestamp}: ${message}\n`;
        debugOutput.scrollTop = debugOutput.scrollHeight;
        debugOutput.style.display = 'block';
      }
    }
  }

  syncSpatialTokensToNode() {
    console.log("=== SYNC SPATIAL TOKENS TO NODE START ===");
    
    if (!this.node) {
      console.log("No node reference available");
      return false;
    }

    // Get current spatial tokens
    const spatialTokensOutput = document.querySelector("#spatialTokensOutput");
    const spatialTokens = spatialTokensOutput ? spatialTokensOutput.value : "";
    
    console.log(`Syncing spatial tokens: ${spatialTokens.length} characters`);
    console.log(`Tokens content: ${spatialTokens}`);

    // Find the spatial_tokens widget
    const spatialTokensWidget = this.node.widgets?.find(w => w.name === "spatial_tokens");
    
    if (spatialTokensWidget) {
      console.log("Found spatial_tokens widget - updating value");
      spatialTokensWidget.value = spatialTokens;
      this.node.setDirtyCanvas(true, true);
      console.log("Spatial tokens synced to Python node");
      return true;
    } else {
      console.log("No spatial_tokens widget found");
      return false;
    }
  }

  sendToNode(node, dialog) {
    console.log("=== SEND TO NODE START ===");
    const tokens = dialog.querySelector("#spatialTokensOutput").value;
    console.log(`Tokens to send: ${tokens.length} characters`);
    console.log(`Tokens content: ${tokens}`);
    
    if (!tokens) {
      console.log("No tokens to send - aborting");
      return;
    }

    console.log(`Node object: ${!!node}, has widgets: ${!!(node && node.widgets)}`);
    
    // Try to find a text widget in the connected node
    if (node && node.widgets) {
      console.log(`Node has ${node.widgets.length} widgets`);
      console.log(`Widget names: ${node.widgets.map(w => w.name).join(', ')}`);
      
      const textWidget = node.widgets.find(
        (w) =>
          w.name === "text" || w.name === "prompt" || w.name === "coordinates",
      );

      if (textWidget) {
        console.log(`Found compatible widget: ${textWidget.name}`);
        console.log(`Widget current value: ${textWidget.value?.length || 0} characters`);
        textWidget.value = tokens;
        console.log(`Widget updated with ${tokens.length} characters`);
        node.setDirtyCanvas(true, true);
        console.log("Node canvas marked as dirty");
        this.updateDebug(
          `Sent tokens to node widget: ${textWidget.name}`,
          dialog,
        );
        console.log("=== SEND TO NODE SUCCESS ===");
      } else {
        console.log("No compatible text widget found");
        this.updateDebug("No compatible text widget found in node", dialog);
        console.log("=== SEND TO NODE FAILED - NO WIDGET ===");
      }
    } else {
      console.log("No node or no widgets available");
      console.log("=== SEND TO NODE FAILED - NO NODE ===");
    }
  }

  addObjectReference(x, y, dialog) {
    const label = dialog.querySelector("#regionLabel").value || "object";

    this.regions.push({
      type: "object_reference",
      label: label,
      coords: [Math.round(x), Math.round(y)],
      normalized: false,
    });

    this.updateRegionsList(dialog);
    this.redrawCanvas();
    this.updateDebug(
      `Added object reference at (${Math.round(x)}, ${Math.round(y)}) with label "${label}"`,
      dialog,
    );
  }

  drawPolygonInProgress() {
    if (this.polygonPoints.length < 1) return;

    // Draw lines between points
    if (this.polygonPoints.length >= 2) {
      this.ctx.strokeStyle = "rgba(0, 255, 255, 0.8)";
      this.ctx.lineWidth = 2;
      this.ctx.setLineDash([5, 5]);

      this.ctx.beginPath();
      const [startX, startY] = this.polygonPoints[0];
      this.ctx.moveTo(startX * this.imageScale, startY * this.imageScale);

      for (let i = 1; i < this.polygonPoints.length; i++) {
        const [x, y] = this.polygonPoints[i];
        this.ctx.lineTo(x * this.imageScale, y * this.imageScale);
      }

      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }

    // Draw snap points with larger circles for better visibility
    this.polygonPoints.forEach(([x, y], index) => {
      const drawX = x * this.imageScale;
      const drawY = y * this.imageScale;

      // Outer circle
      this.ctx.beginPath();
      this.ctx.arc(drawX, drawY, 6, 0, 2 * Math.PI);
      this.ctx.fillStyle = "rgba(0, 255, 255, 0.3)";
      this.ctx.fill();
      this.ctx.strokeStyle = "#00ffff";
      this.ctx.lineWidth = 2;
      this.ctx.stroke();

      // Inner dot
      this.ctx.beginPath();
      this.ctx.arc(drawX, drawY, 2, 0, 2 * Math.PI);
      this.ctx.fillStyle = "#00ffff";
      this.ctx.fill();

      // Point number
      this.ctx.fillStyle = "#00ffff";
      this.ctx.font = "10px system-ui";
      this.ctx.fillText(`${index + 1}`, drawX + 8, drawY - 8);
    });

    // If we have 3+ points, show a preview of closing the polygon
    if (this.polygonPoints.length >= 3) {
      const [firstX, firstY] = this.polygonPoints[0];
      const [lastX, lastY] = this.polygonPoints[this.polygonPoints.length - 1];

      this.ctx.strokeStyle = "rgba(255, 255, 0, 0.6)";
      this.ctx.lineWidth = 1;
      this.ctx.setLineDash([3, 3]);
      this.ctx.beginPath();
      this.ctx.moveTo(lastX * this.imageScale, lastY * this.imageScale);
      this.ctx.lineTo(firstX * this.imageScale, firstY * this.imageScale);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }
  }

  finishPolygon(dialog) {
    if (this.polygonPoints.length < 3) {
      this.updateDebug("Polygon needs at least 3 points", dialog);
      return;
    }

    const label = dialog.querySelector("#regionLabel").value || "polygon";
    const includeObjectRef = dialog.querySelector("#includeObjectRef").checked;

    this.regions.push({
      type: "polygon",
      label: label,
      includeObjectRef: includeObjectRef,
      coords: this.polygonPoints.slice(),
      normalized: false,
    });

    this.polygonPoints = [];
    this.updateRegionsList(dialog);
    this.updateDynamicControls(dialog);
    this.redrawCanvas();
    this.generateTokens(dialog); // Auto-generate tokens
    this.updateDebug(
      `Added polygon with ${this.regions[this.regions.length - 1].coords.length} points, label "${label}" (obj_ref: ${includeObjectRef})`,
      dialog,
    );
  }

  updateDrawingHelp(dialog) {
    const help = {
      bounding_box: "Click and drag to create bounding boxes",
      polygon: "Click to add points. Double-click to finish",
      object_reference: "Click on objects to mark them",
    };
    dialog.querySelector("#drawingHelp").textContent = help[this.drawingMode];
  }

  updateDynamicControls(dialog) {
    const controlsContainer = dialog.querySelector("#dynamicControls");

    if (this.drawingMode === "polygon" && this.polygonPoints.length >= 3) {
      controlsContainer.innerHTML = `
                <button id="finishPolygon" style="
                    width: 100%;
                    padding: 6px;
                    background: rgba(0, 255, 0, 0.2);
                    border: 1px solid #00ff00;
                    border-radius: 4px;
                    color: #00ff00;
                    font-size: 12px;
                    cursor: pointer;
                ">Finish Polygon (${this.polygonPoints.length} points)</button>
            `;

      dialog.querySelector("#finishPolygon").onclick = () => {
        this.finishPolygon(dialog);
      };
    } else {
      controlsContainer.innerHTML = "";
    }
  }

  // Resolution interface handling methods
  updateResolutionInterface(type, dialog) {
    const recommendedDiv = dialog.querySelector('#recommendedResolutions');
    const customDiv = dialog.querySelector('#customResolution');
    
    // Hide all first
    recommendedDiv.style.display = 'none';
    customDiv.style.display = 'none';
    
    if (type === 'recommended') {
      recommendedDiv.style.display = 'block';
      this.populateRecommendedResolutions(dialog);
    } else if (type === 'custom') {
      customDiv.style.display = 'block';
    }
    
    // Auto mode doesn't need any additional UI
  }

  populateRecommendedResolutions(dialog) {
    if (!this.currentImage) {
      console.log("populateRecommendedResolutions: No current image");
      return;
    }
    
    console.log(`populateRecommendedResolutions: originalDimensions=${this.originalDimensions.width}x${this.originalDimensions.height}`);
    console.log(`populateRecommendedResolutions: optimizedDimensions=${this.optimizedDimensions.width}x${this.optimizedDimensions.height}`);
    
    const select = dialog.querySelector('#resolutionSelect');
    if (!select) {
      console.log("populateRecommendedResolutions: No select element found");
      return;
    }
    
    const recommendations = this.getRecommendedResolutions(
      this.originalDimensions.width, 
      this.originalDimensions.height
    );
    
    console.log(`populateRecommendedResolutions: got ${recommendations?.length} recommendations:`, recommendations);
    
    select.innerHTML = recommendations.map(({width, height, label}) => {
      const isCurrentMatch = (width === this.optimizedDimensions.width && height === this.optimizedDimensions.height);
      const aspectRatio = (width / height).toFixed(2);
      const displayLabel = isCurrentMatch ? 
        `${width}x${height} (current) - ${aspectRatio}:1` :
        `${width}x${height} - ${aspectRatio}:1`;
      return `<option value="${width}x${height}">${displayLabel}</option>`;
    }).join('');
    
    console.log(`populateRecommendedResolutions: select.innerHTML length=${select.innerHTML.length}`);
  }

  applyResolution(width, height, dialog) {
    if (!this.currentImage) {
      this.updateDebug("ERROR: No image loaded for resolution change", dialog);
      return;
    }
    
    if (!this.originalImageSrc) {
      this.updateDebug("ERROR: No original image source available for re-optimization", dialog);
      return;
    }
    
    this.updateDebug(`Starting resolution change to ${width}x${height}`, dialog);
    
    // Re-optimize the image with the new target resolution
    const img = new Image();
    img.onload = () => {
      this.updateDebug(`Original image loaded for optimization: ${img.width}x${img.height}`, dialog);
      
      // Get current resize method
      const resizeMethod = dialog.querySelector("#resizeMethod").value;
      this.updateDebug(`Using resize method: ${resizeMethod}`, dialog);
      
      const optimizedData = this.optimizeImageForQwen(img, width, height, resizeMethod);
      
      // Create new optimized image
      this.currentImage = new Image();
      this.currentImage.onload = () => {
        this.updateDebug(`New optimized image loaded: ${this.currentImage.width}x${this.currentImage.height}`, dialog);
        this.updateDebug(`Current optimized dimensions: ${this.optimizedDimensions.width}x${this.optimizedDimensions.height}`, dialog);
        
        // Clear existing regions as coordinates would be invalid
        this.regions = [];
        
        this.updateDebug("Calling updateCanvas...", dialog);
        this.updateCanvas(dialog);
        
        this.updateDebug("Calling updateRegionsList...", dialog);
        this.updateRegionsList(dialog);
        
        this.updateDebug("Calling updateResolutionInfo...", dialog);
        this.updateResolutionInfo(dialog);
        
        this.updateDebug(`=== RESOLUTION CHANGE COMPLETE: ${width}x${height} ===`, dialog);
      };
      
      this.currentImage.onerror = () => {
        this.updateDebug("ERROR: Failed to load optimized image from canvas", dialog);
      };
      
      this.updateDebug(`Setting new image source from optimized canvas`, dialog);
      this.currentImage.src = optimizedData.canvas.toDataURL();
      
      // Store optimization data
      this.optimizationData = optimizedData;
    };
    
    img.onerror = () => {
      this.updateDebug("ERROR: Failed to load original image for re-optimization", dialog);
    };
    
    // Use the original image source to re-optimize
    this.updateDebug(`Loading original image from source for re-optimization`, dialog);
    img.src = this.originalImageSrc;
  }

  updateResolutionInfo(dialog) {
    if (!this.currentImage) return;
    
    const originalSpan = dialog.querySelector('#originalSize');
    const targetSpan = dialog.querySelector('#targetSize');
    const aspectSpan = dialog.querySelector('#aspectRatioInfo');
    
    originalSpan.textContent = `${this.originalDimensions.width}x${this.originalDimensions.height}`;
    targetSpan.textContent = `${this.optimizedDimensions.width}x${this.optimizedDimensions.height}`;
    
    const originalAspect = (this.originalDimensions.width / this.originalDimensions.height).toFixed(2);
    const targetAspect = (this.optimizedDimensions.width / this.optimizedDimensions.height).toFixed(2);
    
    aspectSpan.textContent = `${originalAspect} → ${targetAspect}`;
  }

  updateResizeMethodHelp(method, dialog) {
    const helpElement = dialog.querySelector('#resizeMethodHelp');
    const helpTexts = {
      'pad': 'Scale and add black padding to fit exact dimensions',
      'crop': 'Scale and center crop to fill exact dimensions',
      'stretch': 'Stretch/distort image to exact dimensions (may distort aspect ratio)',
      'resize': 'Scale maintaining aspect ratio (final size may differ from target)'
    };
    helpElement.textContent = helpTexts[method] || '';
  }

  updateDebug(message, dialog) {
    const debugOutput = dialog.querySelector("#debugOutput");
    const timestamp = new Date().toLocaleTimeString();
    debugOutput.textContent += `${timestamp}: ${message}\n`;
    debugOutput.scrollTop = debugOutput.scrollHeight;
  }

  findClosestResolution(width, height) {
    const aspectRatio = width / height;
    
    let bestResolution = null;
    let bestRatioDiff = Infinity;
    
    for (const [resW, resH] of this.QWEN_RESOLUTIONS) {
      const resRatio = resW / resH;
      const ratioDiff = Math.abs(Math.log(resRatio / aspectRatio));
      
      if (ratioDiff < bestRatioDiff) {
        bestRatioDiff = ratioDiff;
        bestResolution = [resW, resH];
      }
    }
    
    return bestResolution;
  }

  getRecommendedResolutions(originalWidth, originalHeight) {
    const aspectRatio = originalWidth / originalHeight;
    
    // Get all resolutions and calculate their aspect ratios and differences
    const recommendations = this.QWEN_RESOLUTIONS.map(([w, h]) => {
      const resRatio = w / h;
      const ratioDiff = Math.abs(Math.log(resRatio / aspectRatio));
      return { width: w, height: h, ratio: resRatio, diff: ratioDiff };
    });
    
    // Sort by aspect ratio similarity (smallest difference first)
    recommendations.sort((a, b) => a.diff - b.diff);
    
    // Return top 5 recommendations
    return recommendations.slice(0, 5).map(r => ({
      width: r.width,
      height: r.height,
      label: `${r.width}x${r.height} (${(r.ratio > 1 ? 'landscape' : r.ratio < 1 ? 'portrait' : 'square')})`,
      isClosest: r === recommendations[0]
    }));
  }

  optimizeImageForQwen(img, targetWidth = null, targetHeight = null, resizeMethod = 'pad') {
    console.log("=== OPTIMIZE IMAGE FOR QWEN START ===");
    console.log(`Input parameters - target: ${targetWidth}x${targetHeight}, method: ${resizeMethod}`);
    console.log(`Input image dimensions: ${img.width}x${img.height}`);
    
    // Store original dimensions
    this.originalDimensions = { width: img.width, height: img.height };
    console.log(`Stored original dimensions: ${this.originalDimensions.width}x${this.originalDimensions.height}`);
    
    // Use provided target resolution or find optimal one
    let targetW, targetH;
    if (targetWidth && targetHeight) {
      targetW = targetWidth;
      targetH = targetHeight;
      console.log(`Using provided target: ${targetW}x${targetH}`);
    } else {
      console.log("No target provided - finding closest Qwen resolution...");
      [targetW, targetH] = this.findClosestResolution(img.width, img.height);
      console.log(`Auto-selected target: ${targetW}x${targetH}`);
    }
    
    this.optimizedDimensions = { width: targetW, height: targetH };
    console.log(`Initial optimized dimensions set: ${targetW}x${targetH}`);
    
    // Create optimized canvas
    console.log("Creating optimized canvas...");
    const optimizedCanvas = document.createElement('canvas');
    optimizedCanvas.width = targetW;
    optimizedCanvas.height = targetH;
    console.log(`Canvas created: ${optimizedCanvas.width}x${optimizedCanvas.height}`);
    const ctx = optimizedCanvas.getContext('2d');
    
    let scale, scaledW, scaledH, offsetX, offsetY;
    console.log(`Applying resize method: ${resizeMethod}`);
    
    switch (resizeMethod) {
      case 'stretch':
        console.log("Applying stretch method - distorting aspect ratio to exact dimensions");
        // Stretch to exact dimensions (distort aspect ratio)
        scale = 1; // Not meaningful for stretch
        scaledW = targetW;
        scaledH = targetH;
        offsetX = 0;
        offsetY = 0;
        console.log(`Stretch parameters - scaledW: ${scaledW}, scaledH: ${scaledH}`);
        ctx.drawImage(img, 0, 0, targetW, targetH);
        break;
        
      case 'crop':
        console.log("Applying crop method - scale to fill and center crop");
        // Scale to fill and center crop
        scale = Math.max(targetW / img.width, targetH / img.height);
        scaledW = img.width * scale;
        scaledH = img.height * scale;
        offsetX = (targetW - scaledW) / 2;
        offsetY = (targetH - scaledH) / 2;
        console.log(`Crop parameters - scale: ${scale}, scaledW: ${scaledW}, scaledH: ${scaledH}, offset: (${offsetX}, ${offsetY})`);
        
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, targetW, targetH);
        ctx.drawImage(img, offsetX, offsetY, scaledW, scaledH);
        break;
        
      case 'resize':
        console.log("Applying resize method - scale maintaining aspect ratio");
        // Scale maintaining aspect ratio (may not fill exact target)
        scale = Math.min(targetW / img.width, targetH / img.height);
        scaledW = img.width * scale;
        scaledH = img.height * scale;
        console.log(`Resize parameters - scale: ${scale}, scaledW: ${scaledW}, scaledH: ${scaledH}`);
        // Update target dimensions to actual final dimensions
        this.optimizedDimensions = { width: Math.round(scaledW), height: Math.round(scaledH) };
        optimizedCanvas.width = Math.round(scaledW);
        optimizedCanvas.height = Math.round(scaledH);
        console.log(`Resize mode: updated canvas to ${optimizedCanvas.width}x${optimizedCanvas.height}`);
        console.log(`Resize mode: updated optimizedDimensions to ${this.optimizedDimensions.width}x${this.optimizedDimensions.height}`);
        offsetX = 0;
        offsetY = 0;
        ctx.drawImage(img, 0, 0, scaledW, scaledH);
        break;
        
      case 'pad':
      default:
        console.log("Applying pad method (default) - scale to fit with black padding");
        // Scale to fit with padding (original behavior)
        scale = Math.min(targetW / img.width, targetH / img.height);
        scaledW = img.width * scale;
        scaledH = img.height * scale;
        offsetX = (targetW - scaledW) / 2;
        offsetY = (targetH - scaledH) / 2;
        console.log(`Pad parameters - scale: ${scale}, scaledW: ${scaledW}, scaledH: ${scaledH}, offset: (${offsetX}, ${offsetY})`);
        
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, targetW, targetH);
        ctx.drawImage(img, offsetX, offsetY, scaledW, scaledH);
        break;
    }
    
    const result = {
      canvas: optimizedCanvas,
      scale: scale,
      offsetX: offsetX,
      offsetY: offsetY,
      targetWidth: this.optimizedDimensions.width,
      targetHeight: this.optimizedDimensions.height,
      resizeMethod: resizeMethod
    };
    
    console.log("Creating optimization result object...");
    console.log(`Result - canvas: ${optimizedCanvas.width}x${optimizedCanvas.height}`);
    console.log(`Result - scale: ${result.scale}, offset: (${result.offsetX}, ${result.offsetY})`);
    console.log(`Result - target dimensions: ${result.targetWidth}x${result.targetHeight}`);
    console.log(`Result - resize method: ${result.resizeMethod}`);
    console.log("=== OPTIMIZE IMAGE FOR QWEN END ===");
    
    return result;
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
}

// Register extension
app.registerExtension({
  name: "Comfy.QwenSpatialInterface",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "QwenSpatialTokenGenerator") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const ret = onNodeCreated
          ? onNodeCreated.apply(this, arguments)
          : undefined;

        const node = this;

        // Expose Python bridge to JavaScript
        if (!window.QwenSpatialTokenGenerator) {
          window.QwenSpatialTokenGenerator = {
            store_optimized_image: (nodeId, base64Data) => {
              console.log(`Bridge: Storing optimized image for node ${nodeId}`);
              // Store in global storage
              window._qwen_spatial_storage = window._qwen_spatial_storage || {};
              window._qwen_spatial_storage[nodeId] = base64Data;
              console.log(`Bridge: Stored in global storage`);
            }
          };
        }

        setTimeout(() => {
          const interfaceBtn = node.addWidget(
            "button",
            "Open Spatial Interface",
            "interface",
            () => {
              const spatialInterface = new QwenSpatialInterface();
              spatialInterface.createInterface(node);
            },
          );
        }, 10);

        return ret;
      };
    }
  },
});
