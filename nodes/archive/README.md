# Archived Nodes

These nodes have been archived because they are broken, superseded, or based on incorrect assumptions.

## Archived Files

### qwen_wan_pure_bridge.py
- **Status**: Incompatible with native ComfyUI
- **Why Archived**: Designed specifically for Kijai's ComfyUI-WanVideoWrapper, not native ComfyUI
- **Issues**: 
  - Returns WANVIDIMAGE_EMBEDS (wrapper-specific type not recognized by native ComfyUI)
  - Uses (C, T, H, W) tensor format without batch dimension (wrapper requirement)
  - Doesn't handle WAN 2.1 vs 2.2 channel differences
- **Note**: Would work with Kijai's wrapper but testing showed quality issues

### qwen_wan_semantic_bridge.py  
- **Status**: Broken approach
- **Why Archived**: Based on incorrect normalization assumptions
- **Issues**:
  - Tried to normalize without understanding channel mismatch
  - "Semantic alignment" doesn't overcome 16ch vs 48ch problem
  - Overly complex for what should be simple

## Why These Failed

The main discovery was that **WAN 2.1 uses 16 channels** (compatible with Qwen) while **WAN 2.2 uses 48 channels** (incompatible). The archived nodes didn't account for this fundamental difference.

## To Re-enable

Set `LOAD_ARCHIVED_NODES = True` in `__init__.py` if you need to access these for reference.