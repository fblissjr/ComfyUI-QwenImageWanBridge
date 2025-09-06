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
  }

  createInterface(node = null) {
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
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        this.currentImage = img;
        this.imageScale = 1;
        this.regions = [];
        this.updateCanvas(dialog);
        this.updateRegionsList(dialog);
        this.updateDebug(`Image loaded: ${img.width}x${img.height}px`, dialog);
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
    this.updateDebug(`generateTokens called: image=${!!this.currentImage}, regions=${this.regions.length}`, dialog);
    
    if (!this.currentImage) {
      dialog.querySelector('#spatialTokensOutput').value = '';
      this.updateDebug("No image loaded - cannot generate tokens", dialog);
      return;
    }
    
    if (this.regions.length === 0) {
      dialog.querySelector('#spatialTokensOutput').value = '';
      this.updateDebug("No regions to generate tokens from", dialog);
      return;
    }
    
    const tokens = this.regions.map(region => {
      if (region.type === 'bounding_box') {
        const [x1, y1, x2, y2] = region.coords;
        const normX1 = (x1 / this.currentImage.width).toFixed(3);
        const normY1 = (y1 / this.currentImage.height).toFixed(3);
        const normX2 = (x2 / this.currentImage.width).toFixed(3);
        const normY2 = (y2 / this.currentImage.height).toFixed(3);
        
        // Make object_ref optional for bounding boxes
        if (region.includeObjectRef !== false) {
          return `<|object_ref_start|>${region.label}<|object_ref_end|> at <|box_start|>${normX1},${normY1},${normX2},${normY2}<|box_end|>`;
        } else {
          return `<|box_start|>${normX1},${normY1},${normX2},${normY2}<|box_end|>`;
        }
        
      } else if (region.type === 'object_reference') {
        // Object reference is just a label, no coordinates
        return `<|object_ref_start|>${region.label}<|object_ref_end|>`;
        
      } else if (region.type === 'polygon') {
        const normalizedPoints = region.coords.map(([x, y]) => 
          `${(x / this.currentImage.width).toFixed(3)},${(y / this.currentImage.height).toFixed(3)}`
        ).join(' ');
        
        // Make object_ref optional for polygons/quads
        if (region.includeObjectRef !== false) {
          return `<|object_ref_start|>${region.label}<|object_ref_end|> outlined by <|quad_start|>${normalizedPoints}<|quad_end|>`;
        } else {
          return `<|quad_start|>${normalizedPoints}<|quad_end|>`;
        }
      }
      
      return ''; // fallback
    }).filter(token => token); // remove empty tokens
    
    const spatialTokens = tokens.join(' ');
    dialog.querySelector('#spatialTokensOutput').value = spatialTokens;
    this.updateDebug(`Generated ${tokens.length} spatial tokens`, dialog);
    
    // Update base_prompt widget with spatial tokens for easy editing
    this.updateBasePromptWithTokens(spatialTokens);
  }

  updateBasePromptWithTokens(spatialTokens) {
    if (!this.node || !this.node.widgets) return;
    
    const basePromptWidget = this.node.widgets.find(w => w.name === 'base_prompt');
    if (!basePromptWidget) return;
    
    // Get current base prompt, removing any existing spatial tokens
    let currentPrompt = basePromptWidget.value || '';
    
    // Remove existing spatial tokens (anything with <|...|> patterns)
    currentPrompt = currentPrompt.replace(/<\|[^|]+\|>/g, '').trim();
    
    // Combine base prompt with spatial tokens
    const combinedPrompt = spatialTokens ? 
      `${currentPrompt} ${spatialTokens}`.trim() : 
      currentPrompt;
      
    basePromptWidget.value = combinedPrompt;
    this.node.setDirtyCanvas(true, true);
  }

  sendToNode(node, dialog) {
    const tokens = dialog.querySelector("#spatialTokensOutput").value;
    if (!tokens) return;

    // Try to find a text widget in the connected node
    if (node && node.widgets) {
      const textWidget = node.widgets.find(
        (w) =>
          w.name === "text" || w.name === "prompt" || w.name === "coordinates",
      );

      if (textWidget) {
        textWidget.value = tokens;
        node.setDirtyCanvas(true, true);
        this.updateDebug(
          `Sent tokens to node widget: ${textWidget.name}`,
          dialog,
        );
      } else {
        this.updateDebug("No compatible text widget found in node", dialog);
      }
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

  updateDebug(message, dialog) {
    const debugOutput = dialog.querySelector("#debugOutput");
    const timestamp = new Date().toLocaleTimeString();
    debugOutput.textContent += `${timestamp}: ${message}\n`;
    debugOutput.scrollTop = debugOutput.scrollHeight;
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
