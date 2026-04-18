/**
 * Qwen Spatial Mask Interface
 * Enhanced spatial interface for mask-based inpainting
 * Extends existing QwenSpatialInterface with 100% compatibility
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

// Import base class (will need to be available globally)
class QwenSpatialMaskInterface {
  constructor() {
    // Inherit ALL existing functionality from QwenSpatialInterface
    this.canvas = null;
    this.ctx = null;
    this.imageCanvas = null;
    this.imageCtx = null;
    this.regions = [];
    this.currentImage = null;
    this.isDrawing = false;
    this.drawingMode = "bounding_box";
    this.outputFormat = "mask_based"; // New default for inpainting
    this.polygonPoints = [];
    this.imageScale = 1;
    this.imageOffset = { x: 0, y: 0 };
    this.node = null;
    
    // NEW: Mask-specific properties
    this.maskCanvas = null;
    this.maskPreviewEnabled = true;
    this.maskProcessingOptions = {
      blur: 2.0,
      expand: 0,
      feather: true,
      pointRadius: 20  // For object references
    };
    
    // Preserve existing Qwen resolutions and optimization data
    this.QWEN_RESOLUTIONS = [
      [1024, 1024],
      [672, 1568], [688, 1504], [720, 1456], [752, 1392],
      [800, 1328], [832, 1248], [880, 1184], [944, 1104],
      [1104, 944], [1184, 880], [1248, 832], [1328, 800],
      [1392, 752], [1456, 720], [1504, 688], [1568, 672],
      [1328, 1328], [1920, 1080], [1080, 1920],
    ];
    
    this.originalDimensions = { width: 0, height: 0 };
    this.optimizedDimensions = { width: 0, height: 0 };
  }

  /**
   * Enhanced interface creation with mask functionality
   * Builds on existing createInterface with mask-specific UI additions
   */
  createInterface(node = null) {
    console.log("=== CREATE MASK INTERFACE START ===");
    console.log(`Node provided: ${!!node}`);
    this.node = node;
    
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
                <h2 style="margin: 0; font-size: 18px; font-weight: 500;">Spatial Mask Editor</h2>
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
                    width: 300px;
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

                    <!-- Drawing Mode -->
                    <div style="margin-bottom: 20px;">
                        <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 500;">Region Type</h4>
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
                            <option value="object_reference">Object Reference Point</option>
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
                            <label style="display: block; margin-bottom: 4px; font-size: 12px; opacity: 0.8;">Editing Instructions:</label>
                            <input type="text" id="regionInstruction" placeholder="edit the object, change to red, remove completely..." style="
                                width: 100%;
                                padding: 6px;
                                background: rgba(255, 255, 255, 0.05);
                                border: 1px solid rgba(255, 255, 255, 0.2);
                                border-radius: 4px;
                                color: white;
                                font-size: 12px;
                            ">
                        </div>

                        <!-- NEW: Point radius for object references -->
                        <div id="pointRadiusSection" style="margin-top: 8px; display: none;">
                            <label style="display: block; margin-bottom: 4px; font-size: 12px; opacity: 0.8;">Point Radius:</label>
                            <input type="range" id="pointRadius" min="5" max="100" value="20" style="width: 100%;">
                            <span id="pointRadiusValue" style="font-size: 11px; opacity: 0.7;">20px</span>
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

                    <!-- NEW: Mask Processing Options -->
                    <div style="margin-bottom: 20px;">
                        <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 500;">Mask Options</h4>
                        
                        <div style="margin-bottom: 8px;">
                            <label style="display: block; margin-bottom: 4px; font-size: 12px; opacity: 0.8;">Blur:</label>
                            <input type="range" id="maskBlur" min="0" max="10" value="2" step="0.5" style="width: 100%;">
                            <span id="maskBlurValue" style="font-size: 11px; opacity: 0.7;">2.0</span>
                        </div>

                        <div style="margin-bottom: 8px;">
                            <label style="display: block; margin-bottom: 4px; font-size: 12px; opacity: 0.8;">Expand/Contract:</label>
                            <input type="range" id="maskExpand" min="-20" max="20" value="0" step="1" style="width: 100%;">
                            <span id="maskExpandValue" style="font-size: 11px; opacity: 0.7;">0px</span>
                        </div>

                        <div style="margin-top: 8px;">
                            <label style="display: flex; align-items: center; font-size: 12px; cursor: pointer;">
                                <input type="checkbox" id="maskFeather" checked style="margin-right: 6px;">
                                Edge Feathering
                            </label>
                        </div>
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
                        <button id="generateMask" style="
                            width: 100%;
                            padding: 10px;
                            background: rgba(28, 150, 75, 0.8);
                            color: white;
                            border: 1px solid rgba(28, 150, 75, 0.5);
                            border-radius: 4px;
                            cursor: pointer;
                            font-weight: 500;
                        ">Generate Mask & Sync</button>
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
                    <h4 style="margin: 0 0 12px 0; font-size: 14px; font-weight: 500;">Mask Preview</h4>

                    <!-- NEW: Mask Preview Canvas -->
                    <div style="
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        border-radius: 4px;
                        padding: 8px;
                        margin-bottom: 16px;
                        background: rgba(255, 255, 255, 0.02);
                        text-align: center;
                    ">
                        <canvas id="maskPreview" style="
                            max-width: 100%;
                            max-height: 200px;
                            border: 1px solid #ccc;
                            background: #f0f0f0;
                        "></canvas>
                        <div style="
                            font-size: 10px;
                            opacity: 0.6;
                            margin-top: 4px;
                        ">White areas = inpaint, Black areas = preserve</div>
                    </div>

                    <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 500;">Inpainting Prompt</h4>

                    <textarea id="inpaintPromptOutput" rows="4" readonly style="
                        width: 100%;
                        padding: 8px;
                        background: rgba(255, 255, 255, 0.05);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        border-radius: 4px;
                        color: white;
                        font-family: system-ui;
                        font-size: 12px;
                        resize: vertical;
                        margin-bottom: 8px;
                    " placeholder="Generated inpainting instructions will appear here..."></textarea>

                    <div style="display: flex; gap: 6px; margin-bottom: 16px;">
                        <button id="copyPrompt" style="
                            flex: 1;
                            padding: 6px;
                            background: rgba(255, 255, 255, 0.1);
                            color: white;
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 12px;
                        ">Copy</button>
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
    this.updateMaskProcessingOptionsUI(dialog);
    
    // Try to auto-load connected image
    setTimeout(() => {
      this.loadImageFromConnectedNode(dialog);
    }, 500);
    
    return dialog;
  }

  /**
   * Core mask generation - converts spatial regions to binary mask
   * BLACK = preserve areas, WHITE = edit areas (follows diffusers convention)
   */
  generateMask(dialog) {
    console.log("=== GENERATE MASK START ===");
    
    if (!this.currentImage || this.regions.length === 0) {
      console.log("No image or regions available for mask generation");
      if (dialog) {
        this.updateDebug("No image or regions - cannot generate mask", dialog);
      }
      return null;
    }
    
    const imageWidth = this.optimizedDimensions.width;
    const imageHeight = this.optimizedDimensions.height;
    
    console.log(`Generating mask for dimensions: ${imageWidth}x${imageHeight}`);
    console.log(`Processing ${this.regions.length} regions`);
    
    // Create mask canvas
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = imageWidth;
    maskCanvas.height = imageHeight;
    const ctx = maskCanvas.getContext('2d');
    
    // Initialize as fully preserved (black background)
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, imageWidth, imageHeight);
    
    // Paint edit regions (white)
    ctx.fillStyle = '#FFFFFF';
    
    // Use existing regions array - NO CHANGES to region handling
    this.regions.forEach((region, index) => {
      console.log(`Processing region ${index}: ${region.type} "${region.label}"`);
      this.paintRegionToMask(ctx, region);
    });
    
    this.maskCanvas = maskCanvas;
    
    // Only update preview if dialog is available
    if (dialog) {
      this.updateMaskPreview(dialog);
    }
    
    console.log("Mask generation complete");
    console.log("=== GENERATE MASK END ===");
    
    return maskCanvas.toDataURL('image/png');
  }

  /**
   * Paint different region types to mask canvas
   * Uses existing coordinate systems - NO CHANGES needed
   */
  paintRegionToMask(ctx, region) {
    switch(region.type) {
      case 'bounding_box':
        const [x1, y1, x2, y2] = region.coords;
        ctx.fillRect(
          Math.min(x1, x2),
          Math.min(y1, y2),
          Math.abs(x2 - x1),
          Math.abs(y2 - y1)
        );
        console.log(`Painted bounding box: (${x1},${y1}) to (${x2},${y2})`);
        break;
        
      case 'polygon':
        if (region.coords.length >= 3) {
          ctx.beginPath();
          ctx.moveTo(region.coords[0][0], region.coords[0][1]);
          region.coords.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
          ctx.closePath();
          ctx.fill();
          console.log(`Painted polygon with ${region.coords.length} points`);
        }
        break;
        
      case 'object_reference':
        const [x, y] = region.coords;
        const radius = this.maskProcessingOptions.pointRadius || 20;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fill();
        console.log(`Painted object reference at (${x},${y}) with radius ${radius}`);
        break;
    }
  }

  /**
   * Enhanced sync function - sends mask + prompt to Python node
   * Builds on existing widget sync architecture
   */
  syncToNode() {
    console.log("=== SYNC TO NODE START ===");
    
    if (!this.node) {
      console.log("No node reference available");
      return false;
    }

    // Generate and send mask
    const maskData = this.generateMask();
    if (!maskData) {
      console.log("Failed to generate mask data");
      return false;
    }
    
    const maskWidget = this.node.widgets?.find(w => w.name === "mask_data");
    if (maskWidget) {
      maskWidget.value = maskData;
      console.log("Mask data synced to node");
    } else {
      console.log("No mask_data widget found");
    }
    
    // Generate and send inpainting prompt
    const prompt = this.generateInpaintPrompt();
    const promptWidget = this.node.widgets?.find(w => w.name === "inpaint_prompt");  
    if (promptWidget) {
      promptWidget.value = prompt;
      console.log("Inpaint prompt synced to node");
    } else {
      console.log("No inpaint_prompt widget found");
    }
    
    // Send processing options
    const optionsWidget = this.node.widgets?.find(w => w.name === "mask_options");
    if (optionsWidget) {
      optionsWidget.value = JSON.stringify(this.maskProcessingOptions);
      console.log("Mask options synced to node");
    } else {
      console.log("No mask_options widget found");
    }
    
    // Use existing node sync pattern
    this.node.setDirtyCanvas(true, true);
    
    console.log("=== SYNC TO NODE END ===");
    return true;
  }

  /**
   * Generate contextual inpainting prompts from regions
   * Uses existing region.label and adds instruction field
   */
  generateInpaintPrompt() {
    if (this.regions.length === 0) return "";
    
    const regionPrompts = this.regions.map(region => {
      const target = region.label || "area";
      const instruction = region.instruction || `edit the ${target}`;
      return instruction;
    });
    
    return regionPrompts.join(', ');
  }

  /**
   * Real-time mask preview
   * Shows user exactly what areas will be inpainted
   */
  updateMaskPreview(dialog) {
    // Safety check - ensure dialog exists
    if (!dialog) {
      console.warn("updateMaskPreview: dialog is undefined, skipping preview update");
      return;
    }
    
    const preview = dialog.querySelector('#maskPreview');
    if (preview && this.maskCanvas) {
      const previewCtx = preview.getContext('2d');
      const scale = Math.min(280/this.maskCanvas.width, 200/this.maskCanvas.height);
      
      preview.width = this.maskCanvas.width * scale;
      preview.height = this.maskCanvas.height * scale;
      
      previewCtx.drawImage(this.maskCanvas, 0, 0, preview.width, preview.height);
      console.log(`Mask preview updated: ${preview.width}x${preview.height}`);
    }
  }

  /**
   * Update mask processing options UI
   */
  updateMaskProcessingOptionsUI(dialog) {
    // Update blur value display
    const blurSlider = dialog.querySelector('#maskBlur');
    const blurValue = dialog.querySelector('#maskBlurValue');
    if (blurSlider && blurValue) {
      blurSlider.oninput = (e) => {
        this.maskProcessingOptions.blur = parseFloat(e.target.value);
        blurValue.textContent = e.target.value;
      };
    }
    
    // Update expand value display
    const expandSlider = dialog.querySelector('#maskExpand');
    const expandValue = dialog.querySelector('#maskExpandValue');
    if (expandSlider && expandValue) {
      expandSlider.oninput = (e) => {
        this.maskProcessingOptions.expand = parseInt(e.target.value);
        expandValue.textContent = e.target.value + "px";
      };
    }
    
    // Update point radius display
    const radiusSlider = dialog.querySelector('#pointRadius');
    const radiusValue = dialog.querySelector('#pointRadiusValue');
    if (radiusSlider && radiusValue) {
      radiusSlider.oninput = (e) => {
        this.maskProcessingOptions.pointRadius = parseInt(e.target.value);
        radiusValue.textContent = e.target.value + "px";
      };
    }
    
    // Update feather checkbox
    const featherCheck = dialog.querySelector('#maskFeather');
    if (featherCheck) {
      featherCheck.onchange = (e) => {
        this.maskProcessingOptions.feather = e.target.checked;
      };
    }
  }

  // ============================================================================
  // ALL EXISTING METHODS FROM QwenSpatialInterface PRESERVED
  // These maintain 100% compatibility with existing functionality
  // ============================================================================

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
      this.updatePointRadiusVisibility(dialog);
      this.redrawCanvas();
    };

    // Debug mode toggle
    dialog.querySelector("#debugMode").onchange = (e) => {
      const debugOutput = dialog.querySelector("#debugOutput");
      debugOutput.style.display = e.target.checked ? "block" : "none";
    };

    // Clear regions
    dialog.querySelector("#clearRegions").onclick = () => {
      this.regions = [];
      this.updateRegionsList(dialog);
      this.redrawCanvas();
      this.generateAndDisplayMask(dialog);
    };

    // NEW: Generate mask instead of tokens
    dialog.querySelector("#generateMask").onclick = () => {
      this.generateAndDisplayMask(dialog);
      this.syncToNode();
    };

    // Copy prompt
    dialog.querySelector("#copyPrompt").onclick = () => {
      const prompt = dialog.querySelector("#inpaintPromptOutput").value;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(prompt);
        this.updateDebug("Prompt copied to clipboard", dialog);
      } else {
        // Fallback for older browsers
        const textArea = document.createElement("textarea");
        textArea.value = prompt;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand("copy");
        document.body.removeChild(textArea);
        this.updateDebug("Prompt copied to clipboard (fallback)", dialog);
      }
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

  updatePointRadiusVisibility(dialog) {
    const section = dialog.querySelector('#pointRadiusSection');
    if (section) {
      section.style.display = this.drawingMode === 'object_reference' ? 'block' : 'none';
    }
  }

  generateAndDisplayMask(dialog) {
    const maskData = this.generateMask(dialog);
    const prompt = this.generateInpaintPrompt();
    
    // Update UI
    dialog.querySelector('#inpaintPromptOutput').value = prompt;
    
    this.updateDebug(`Generated mask with ${this.regions.length} regions`, dialog);
    this.updateDebug(`Inpaint prompt: ${prompt}`, dialog);
  }

  // ALL EXISTING METHODS FROM ORIGINAL QwenSpatialInterface
  // (loadImage, setupCanvas, setupCanvasEvents, updateCanvas, redrawCanvas, etc.)
  // These are preserved exactly to maintain 100% compatibility
  
  loadImage(file, dialog) {
    console.log("=== SPATIAL MASK INTERFACE: LOAD IMAGE START ===");
    console.log(`Loading file: ${file.name}, size: ${file.size} bytes, type: ${file.type}`);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      console.log(`FileReader loaded: ${e.target.result.length} characters`);
      this.originalImageSrc = e.target.result;
      
      const img = new Image();
      img.onload = () => {
        console.log(`Original image loaded: ${img.width}x${img.height}`);
        
        // Optimize the image for Qwen coordinate system
        console.log("Starting image optimization for Qwen...");
        const optimizedData = this.optimizeImageForQwen(img, null, null, 'pad');
        
        // Create new image from optimized canvas
        this.currentImage = new Image();
        this.currentImage.onload = () => {
          console.log(`Optimized image loaded: ${this.currentImage.width}x${this.currentImage.height}`);
          
          this.imageScale = 1;
          this.regions = [];
          this.updateCanvas(dialog);
          this.updateRegionsList(dialog);
          this.updateDebug(`Image loaded and optimized for mask editing`, dialog);
          this.updateDebug(`Original: ${this.originalDimensions.width}x${this.originalDimensions.height}px`, dialog);
          this.updateDebug(`Optimized: ${this.optimizedDimensions.width}x${this.optimizedDimensions.height}px`, dialog);
          
          console.log("=== SPATIAL MASK INTERFACE: LOAD IMAGE COMPLETE ===");
        };
        this.currentImage.src = optimizedData.canvas.toDataURL();
        this.optimizationData = optimizedData;
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
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
    let startX, startY, isDragging = false;

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

  updateCanvas(dialog) {
    if (!this.currentImage) return;

    this.imageCtx.clearRect(0, 0, this.imageCanvas.width, this.imageCanvas.height);

    const imgWidth = this.currentImage.width * this.imageScale;
    const imgHeight = this.currentImage.height * this.imageScale;
    this.imageCtx.drawImage(this.currentImage, 0, 0, imgWidth, imgHeight);

    dialog.querySelector("#zoomLevel").textContent = `${Math.round(this.imageScale * 100)}%`;
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

      this.ctx.beginPath();
      this.ctx.arc(drawX, drawY, 8, 0, 2 * Math.PI);
      this.ctx.fillStyle = color;
      this.ctx.fill();
      this.ctx.stroke();

      this.ctx.beginPath();
      this.ctx.arc(drawX, drawY, 3, 0, 2 * Math.PI);
      this.ctx.fillStyle = "white";
      this.ctx.fill();

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

        region.coords.forEach(([x, y]) => {
          const drawX = x * this.imageScale;
          const drawY = y * this.imageScale;

          this.ctx.beginPath();
          this.ctx.arc(drawX, drawY, 4, 0, 2 * Math.PI);
          this.ctx.fillStyle = color;
          this.ctx.fill();
        });

        const [labelX, labelY] = region.coords[0];
        this.ctx.fillStyle = color;
        this.ctx.font = "12px system-ui";
        this.ctx.fillText(region.label, labelX * this.imageScale, labelY * this.imageScale - 15);
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
    const instruction = dialog.querySelector("#regionInstruction").value || `edit the ${label}`;

    const region = {
      type: "bounding_box",
      label: label,
      instruction: instruction,
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
    this.generateAndDisplayMask(dialog);
    this.updateDebug(`Added bounding box: ${label}`, dialog);
  }

  addObjectReference(x, y, dialog) {
    const label = dialog.querySelector("#regionLabel").value || "object";
    const instruction = dialog.querySelector("#regionInstruction").value || `focus on the ${label}`;

    this.regions.push({
      type: "object_reference",
      label: label,
      instruction: instruction,
      coords: [Math.round(x), Math.round(y)],
    });

    this.updateRegionsList(dialog);
    this.redrawCanvas();
    this.generateAndDisplayMask(dialog);
    this.updateDebug(`Added object reference at (${Math.round(x)}, ${Math.round(y)}) with label "${label}"`, dialog);
  }

  finishPolygon(dialog) {
    if (this.polygonPoints.length < 3) {
      this.updateDebug("Polygon needs at least 3 points", dialog);
      return;
    }

    const label = dialog.querySelector("#regionLabel").value || "polygon";
    const instruction = dialog.querySelector("#regionInstruction").value || `edit the ${label}`;

    this.regions.push({
      type: "polygon",
      label: label,
      instruction: instruction,
      coords: this.polygonPoints.slice(),
    });

    this.polygonPoints = [];
    this.updateRegionsList(dialog);
    this.updateDynamicControls(dialog);
    this.redrawCanvas();
    this.generateAndDisplayMask(dialog);
    this.updateDebug(`Added polygon with ${this.regions[this.regions.length - 1].coords.length} points, label "${label}"`, dialog);
  }

  drawPolygonInProgress() {
    if (this.polygonPoints.length < 1) return;

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

    this.polygonPoints.forEach(([x, y], index) => {
      const drawX = x * this.imageScale;
      const drawY = y * this.imageScale;

      this.ctx.beginPath();
      this.ctx.arc(drawX, drawY, 6, 0, 2 * Math.PI);
      this.ctx.fillStyle = "rgba(0, 255, 255, 0.3)";
      this.ctx.fill();
      this.ctx.strokeStyle = "#00ffff";
      this.ctx.lineWidth = 2;
      this.ctx.stroke();

      this.ctx.beginPath();
      this.ctx.arc(drawX, drawY, 2, 0, 2 * Math.PI);
      this.ctx.fillStyle = "#00ffff";
      this.ctx.fill();

      this.ctx.fillStyle = "#00ffff";
      this.ctx.font = "10px system-ui";
      this.ctx.fillText(`${index + 1}`, drawX + 8, drawY - 8);
    });

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
                            <div style="color: ${color}; font-weight: 500; font-size: 12px;">${region.label}</div>
                            <div style="opacity: 0.7; font-size: 10px;">${region.type}</div>
                            <div style="opacity: 0.8; font-size: 10px; margin-top: 2px;">${region.instruction || 'No instruction'}</div>
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
    this.setupRegionListEvents(dialog);
  }

  setupRegionListEvents(dialog) {
    dialog.querySelectorAll('.region-delete-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const index = parseInt(e.target.dataset.regionIndex);
        this.deleteRegion(index, dialog);
      });
    });
  }

  deleteRegion(index, dialog) {
    if (index < 0 || index >= this.regions.length) return;
    
    const deletedRegion = this.regions[index];
    
    if (!confirm(`Delete region "${deletedRegion.label}"?`)) return;
    
    this.regions.splice(index, 1);
    
    this.updateRegionsList(dialog);
    this.redrawCanvas();
    this.generateAndDisplayMask(dialog);
    
    this.updateDebug(`Deleted region: ${deletedRegion.label} (${deletedRegion.type})`, dialog);
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

  loadImageFromConnectedNode(dialog) {
    console.log("=== LOAD IMAGE FROM CONNECTED NODE START ===");
    
    if (!this.node) {
      console.log("No node reference available");
      return;
    }

    const imageInput = this.node.inputs?.find(input => input.name === "image");
    
    if (!imageInput || !imageInput.link) {
      console.log("No image input connected to node");
      this.updateDebug("No image connected - please connect an image to the node and upload it in this interface", dialog);
      return;
    }

    console.log(`Found connected image input: ${imageInput.name}`);
    
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
    
    const sourceInfo = `${sourceNode.title || sourceNode.type}`;
    this.updateDebug(`Image connected from: ${sourceInfo}`, dialog);
    this.updateDebug(`Please upload the same image in this interface for mask editing`, dialog);
    this.updateDebug(`The mask will be automatically applied to the connected image`, dialog);
    
    console.log("=== LOAD IMAGE FROM CONNECTED NODE END ===");
  }

  updateDebug(message, dialog) {
    // Safety check - ensure dialog exists
    if (!dialog) {
      console.warn("updateDebug: dialog is undefined, logging to console instead:", message);
      return;
    }
    
    const debugOutput = dialog.querySelector("#debugOutput");
    if (!debugOutput) {
      console.warn("updateDebug: debugOutput element not found, logging to console instead:", message);
      return;
    }
    
    const timestamp = new Date().toLocaleTimeString();
    debugOutput.textContent += `${timestamp}: ${message}\n`;
    debugOutput.scrollTop = debugOutput.scrollHeight;
  }

  // Preserve all existing utility methods
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

  optimizeImageForQwen(img, targetWidth = null, targetHeight = null, resizeMethod = 'pad') {
    console.log("=== OPTIMIZE IMAGE FOR QWEN START ===");
    
    this.originalDimensions = { width: img.width, height: img.height };
    
    let targetW, targetH;
    if (targetWidth && targetHeight) {
      targetW = targetWidth;
      targetH = targetHeight;
    } else {
      [targetW, targetH] = this.findClosestResolution(img.width, img.height);
    }
    
    this.optimizedDimensions = { width: targetW, height: targetH };
    
    const optimizedCanvas = document.createElement('canvas');
    optimizedCanvas.width = targetW;
    optimizedCanvas.height = targetH;
    const ctx = optimizedCanvas.getContext('2d');
    
    // Apply padding method (default for mask generation)
    const scale = Math.min(targetW / img.width, targetH / img.height);
    const scaledW = img.width * scale;
    const scaledH = img.height * scale;
    const offsetX = (targetW - scaledW) / 2;
    const offsetY = (targetH - scaledH) / 2;
    
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, targetW, targetH);
    ctx.drawImage(img, offsetX, offsetY, scaledW, scaledH);
    
    console.log("=== OPTIMIZE IMAGE FOR QWEN END ===");
    
    return {
      canvas: optimizedCanvas,
      scale: scale,
      offsetX: offsetX,
      offsetY: offsetY,
      targetWidth: this.optimizedDimensions.width,
      targetHeight: this.optimizedDimensions.height,
      resizeMethod: resizeMethod
    };
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

// Export for use by other modules
export { QwenSpatialMaskInterface };

// Register extension for mask-based nodes
app.registerExtension({
  name: "Comfy.QwenSpatialMaskInterface",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // Register for both existing QwenSpatialTokenGenerator and new QwenMaskProcessor
    if (nodeData.name === "QwenSpatialTokenGenerator" || nodeData.name === "QwenMaskProcessor") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        const node = this;

        setTimeout(() => {
          const interfaceBtn = node.addWidget(
            "button",
            "Open Mask Editor",
            "interface",
            () => {
              const spatialMaskInterface = new QwenSpatialMaskInterface();
              spatialMaskInterface.createInterface(node);
            },
          );
        }, 10);

        return ret;
      };
    }
  },
});