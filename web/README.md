# Qwen ComfyUI Web Extensions

This directory contains JavaScript extensions for ComfyUI that provide advanced tokenization analysis and spatial editing tools for Qwen2.5-VL models.

## Files Overview

### Core Extensions

#### `qwen_template_builder.js`
- **Purpose**: Template system integration for QwenTemplateBuilder node
- **Features**:
  - Auto-updates system prompts based on preset selection
  - Manages vision token inclusion settings
  - Simple dropdown-driven interface

#### `qwen_token_analyzer.js` 
- **Purpose**: Token analysis and spatial coordinate editing
- **Features**:
  - Real-time token sequence analysis
  - Interactive spatial coordinate editor with canvas
  - Template prompt generation
  - Token validation and error detection
- **Adds to nodes**: QwenVLTextEncoder

#### `qwen_token_visualizer.js`
- **Purpose**: Advanced token sequence visualization
- **Features**:
  - Color-coded token display by category
  - Sequence highlighting and grouping
  - Interactive token selection
  - Export analysis to JSON
  - Token efficiency metrics
- **Adds to nodes**: QwenTokenDebugger

#### `qwen_testing_interface.js`
- **Purpose**: Comprehensive testing environment
- **Features**:
  - Pre-built test templates for different use cases
  - Custom test creation
  - Validation rules for coordinates and sequences  
  - Batch testing and result export
  - Error detection and reporting
- **Adds to nodes**: QwenVLTextEncoder, QwenTokenDebugger

## Token Categories

The extensions recognize and handle these Qwen2.5-VL token categories:

### Vision Processing Tokens
- `<|vision_start|>` (ID: 151652) - Begin vision processing
- `<|vision_end|>` (ID: 151653) - End vision processing  
- `<|image_pad|>` (ID: 151655) - Image patch placeholder
- `<|video_pad|>` (ID: 151656) - Video frame placeholder
- `<|vision_pad|>` (ID: 151654) - General vision padding

### Spatial Reference Tokens
- `<|object_ref_start|>` / `<|object_ref_end|>` (ID: 151646-151647) - Object identification
- `<|box_start|>` / `<|box_end|>` (ID: 151648-151649) - Bounding box coordinates
- `<|quad_start|>` / `<|quad_end|>` (ID: 151650-151651) - Polygon coordinates

### Chat Format Tokens
- `<|im_start|>` (ID: 151644) - Begin chat message
- `<|im_end|>` (ID: 151645) - End chat message

### Code Completion Tokens
- `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>` - Fill-in-the-middle
- `<|repo_name|>`, `<|file_sep|>` - Repository context

### Tool Calling Tokens
- `<tool_call>` / `</tool_call>` - Function calling markers

## Usage Guide

### 1. Token Analysis (qwen_token_analyzer.js)

When using QwenVLTextEncoder nodes, you'll see an "Analyze Tokens" button:

```javascript
// Click "Analyze Tokens" to get:
// - Token count statistics
// - Sequence validation
// - Template suggestions
// - Error detection
```

### 2. Spatial Coordinate Editor

Click "Spatial Editor" to open an interactive canvas:

```javascript
// Features:
// - Click and drag to create bounding boxes
// - Automatic token generation: <|box_start|>x1,y1,x2,y2<|box_end|>
// - Multiple region support
// - Export tokens to clipboard
```

### 3. Token Visualization (qwen_token_visualizer.js)

The QwenTokenDebugger node automatically shows a visual token breakdown:

```javascript
// Visual features:
// - Color-coded tokens by category
// - Sequence highlighting
// - Click tokens to highlight/unhighlight
// - Export analysis as JSON
```

### 4. Testing Interface (qwen_testing_interface.js)

Click "Open Testing Interface" for comprehensive testing:

```javascript
// Test templates included:
// - Basic Vision: <|vision_start|><|image_pad|><|vision_end|>
// - Chat + Vision: Full conversation format
// - Spatial Editing: Object references with coordinates
// - Multi-object: Multiple spatial references
// - Code Completion: Fill-in-the-middle patterns
```

## Common Usage Patterns

### Basic Vision Processing
```
Describe this image: <|vision_start|><|image_pad|><|vision_end|>
```

### Chat with Vision  
```
<|im_start|>user
Analyze this image: <|vision_start|><|image_pad|><|vision_end|>
What objects do you see?
<|im_end|>
```

### Spatial Object Editing
```
<|im_start|>user
Edit the <|object_ref_start|>car<|object_ref_end|> at <|box_start|>100,50,300,200<|box_end|> 
in this image: <|vision_start|><|image_pad|><|vision_end|>
Make it red.
<|im_end|>
```

### Multiple Object References
```
Edit the <|object_ref_start|>car<|object_ref_end|> at <|box_start|>100,100,200,200<|box_end|> 
and the <|object_ref_start|>tree<|object_ref_end|> at <|box_start|>300,50,400,150<|box_end|>
```

### Polygon-based References
```
Modify the building outline <|quad_start|>10,20 100,25 95,80 8,75<|quad_end|> to add windows.
```

## Coordinate Formats

### Bounding Box Coordinates
- Format: `<|box_start|>x1,y1,x2,y2<|box_end|>`
- Example: `<|box_start|>100,50,300,200<|box_end|>`
- Validation: x2 > x1 and y2 > y1

### Polygon Coordinates  
- Format: `<|quad_start|>x1,y1 x2,y2 x3,y3 x4,y4<|quad_end|>`
- Example: `<|quad_start|>10,20 100,25 95,80 8,75<|quad_end|>`
- Validation: Space-separated x,y coordinate pairs

## Integration with Nodes

### QwenVLTextEncoder
- Adds "Analyze Tokens" button
- Adds "Spatial Editor" button
- Adds "Open Testing Interface" button
- All features work with the text/prompt input

### QwenTokenDebugger  
- Automatically displays token visualizer
- Shows real-time analysis as you type
- Provides validation and error reporting
- Adds "Open Testing Interface" button

### QwenTemplateBuilder
- Auto-updates system prompts based on presets
- Manages vision token inclusion
- Simple dropdown interface

## Error Detection

The extensions automatically detect common issues:

### Coordinate Errors
- Invalid number format in coordinates
- Wrong number of coordinate values
- Malformed coordinate pairs

### Sequence Errors
- Unmatched opening/closing tokens
- Empty vision sequences
- Malformed token structures

### Validation Messages
- Clear error descriptions
- Position information for errors
- Suggested corrections

## Export Features

### Token Analysis Export
- JSON format with complete analysis
- Token breakdown by category
- Sequence information
- Efficiency metrics

### Test Results Export
- Comprehensive test history
- Custom test definitions
- Validation results
- Performance metrics

## Browser Compatibility

- Modern browsers with ES6+ support
- Canvas API for spatial editor
- Local storage for preferences
- Clipboard API for copy/paste

## Performance Notes

- Token analysis is performed client-side
- Real-time updates for small texts
- Debounced analysis for large inputs  
- Efficient regex-based parsing
- Minimal DOM manipulation

## Troubleshooting

### Common Issues

1. **Buttons not appearing**
   - Ensure ComfyUI is fully loaded
   - Check browser console for errors
   - Verify web directory is properly configured

2. **Spatial editor not working**
   - Check Canvas API support
   - Verify mouse event handling
   - Clear browser cache

3. **Token analysis errors**
   - Validate input text format
   - Check for unsupported token combinations
   - Review coordinate formatting

### Debug Mode

Enable debug logging in browser console:
```javascript
localStorage.setItem('qwen_debug', 'true');
```

This enables detailed logging of token analysis and validation processes.