#!/usr/bin/env python3
"""
Qwen2.5-VL Tokenizer Analysis Tool
Analyzes the special tokens used in Qwen2.5-VL for vision and language processing.
"""

import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict

class QwenTokenAnalyzer:
    def __init__(self, tokenizer_path: str = "/Users/fredbliss/Storage/Qwen-Image-Edit"):
        self.tokenizer_path = tokenizer_path
        self.added_tokens = {}
        self.config = {}
        self.special_tokens = {}
        self.load_configs()
    
    def load_configs(self):
        """Load tokenizer configuration files"""
        # Load added tokens
        added_tokens_path = os.path.join(self.tokenizer_path, "tokenizer", "added_tokens.json")
        if os.path.exists(added_tokens_path):
            with open(added_tokens_path, 'r') as f:
                self.added_tokens = json.load(f)
        
        # Load tokenizer config
        config_path = os.path.join(self.tokenizer_path, "tokenizer", "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Load special tokens map if exists
        special_path = os.path.join(self.tokenizer_path, "tokenizer", "special_tokens_map.json")
        if os.path.exists(special_path):
            with open(special_path, 'r') as f:
                self.special_tokens = json.load(f)
    
    def categorize_tokens(self) -> Dict[str, List[Tuple[str, int]]]:
        """Categorize tokens by their apparent function"""
        categories = defaultdict(list)
        
        for token, token_id in self.added_tokens.items():
            # Vision-related tokens
            if any(x in token.lower() for x in ['vision', 'image', 'video']):
                categories['Vision Processing'].append((token, token_id))
            # Spatial reference tokens
            elif any(x in token.lower() for x in ['box', 'quad', 'object_ref']):
                categories['Spatial References'].append((token, token_id))
            # Chat/conversation tokens
            elif any(x in token.lower() for x in ['im_start', 'im_end']):
                categories['Chat Format'].append((token, token_id))
            # Code completion tokens
            elif any(x in token.lower() for x in ['fim_', 'repo_', 'file_']):
                categories['Code Completion'].append((token, token_id))
            # Tool calling tokens
            elif 'tool_call' in token.lower():
                categories['Tool Calling'].append((token, token_id))
            # Special control tokens
            elif any(x in token.lower() for x in ['endoftext', 'pad']):
                categories['Control Tokens'].append((token, token_id))
            else:
                categories['Other'].append((token, token_id))
        
        return dict(categories)
    
    def analyze_vision_tokens(self) -> Dict[str, str]:
        """Analyze vision-specific tokens and their likely purposes"""
        vision_analysis = {}
        
        vision_tokens = {
            "<|vision_start|>": "Marks the beginning of vision processing in the input sequence",
            "<|vision_end|>": "Marks the end of vision processing in the input sequence", 
            "<|vision_pad|>": "Padding token used within vision sequences for alignment",
            "<|image_pad|>": "Specific padding for image patches/tokens during vision processing",
            "<|video_pad|>": "Specific padding for video frames/tokens during vision processing"
        }
        
        for token, description in vision_tokens.items():
            if token in self.added_tokens:
                vision_analysis[token] = {
                    'token_id': self.added_tokens[token],
                    'purpose': description,
                    'is_special': self.is_special_token(token)
                }
        
        return vision_analysis
    
    def analyze_spatial_tokens(self) -> Dict[str, str]:
        """Analyze spatial reference tokens for object detection/grounding"""
        spatial_analysis = {}
        
        spatial_tokens = {
            "<|object_ref_start|>": "Begins reference to a specific object in the image",
            "<|object_ref_end|>": "Ends reference to a specific object in the image",
            "<|box_start|>": "Begins bounding box coordinates (likely x1,y1,x2,y2 format)",
            "<|box_end|>": "Ends bounding box coordinates",
            "<|quad_start|>": "Begins quadrilateral/polygon coordinates for irregular shapes",
            "<|quad_end|>": "Ends quadrilateral/polygon coordinates"
        }
        
        for token, description in spatial_tokens.items():
            if token in self.added_tokens:
                spatial_analysis[token] = {
                    'token_id': self.added_tokens[token],
                    'purpose': description,
                    'is_special': self.is_special_token(token)
                }
        
        return spatial_analysis
    
    def is_special_token(self, token: str) -> bool:
        """Check if a token is marked as special in the configuration"""
        if 'added_tokens_decoder' in self.config:
            for token_info in self.config['added_tokens_decoder'].values():
                if token_info.get('content') == token:
                    return token_info.get('special', False)
        return False
    
    def generate_test_prompts(self) -> List[Dict[str, str]]:
        """Generate test prompts to understand token behavior"""
        test_cases = [
            {
                'name': 'Basic Vision Processing',
                'prompt': 'Describe what you see in this image: <|vision_start|><|image_pad|><|vision_end|>',
                'purpose': 'Test basic vision token sequence'
            },
            {
                'name': 'Object Reference',
                'prompt': 'The <|object_ref_start|>red car<|object_ref_end|> is located at <|box_start|>100,50,200,150<|box_end|>',
                'purpose': 'Test object grounding with bounding box'
            },
            {
                'name': 'Spatial Editing',
                'prompt': 'Edit the region <|box_start|>0,0,100,100<|box_end|> to make it brighter',
                'purpose': 'Test spatial editing capabilities'
            },
            {
                'name': 'Complex Spatial Reference',
                'prompt': 'The building outline follows <|quad_start|>10,20 100,25 95,80 8,75<|quad_end|>',
                'purpose': 'Test polygon-based spatial references'
            },
            {
                'name': 'Multi-Modal Chat',
                'prompt': '<|im_start|>user\nAnalyze this image: <|vision_start|><|image_pad|><|vision_end|>\nWhat objects do you see?<|im_end|>',
                'purpose': 'Test chat format with vision integration'
            }
        ]
        
        return test_cases
    
    def create_exploration_interface(self) -> str:
        """Generate HTML interface for token exploration"""
        html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen2.5-VL Token Explorer</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-8 text-center text-gray-800">Qwen2.5-VL Token Explorer</h1>
        
        <!-- Token Categories -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {token_cards}
        </div>
        
        <!-- Test Prompt Generator -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-2xl font-bold mb-4">Test Prompt Generator</h2>
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Select Token Type:</label>
                    <select id="tokenType" class="w-full p-2 border border-gray-300 rounded-md">
                        <option value="vision">Vision Processing</option>
                        <option value="spatial">Spatial References</option>
                        <option value="chat">Chat Format</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Custom Text:</label>
                    <textarea id="customText" rows="3" class="w-full p-2 border border-gray-300 rounded-md" 
                              placeholder="Enter your prompt text here..."></textarea>
                </div>
                <button onclick="generatePrompt()" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                    Generate Test Prompt
                </button>
                <div id="generatedPrompt" class="mt-4 p-4 bg-gray-100 rounded-md hidden">
                    <h3 class="font-bold mb-2">Generated Prompt:</h3>
                    <pre class="whitespace-pre-wrap text-sm"></pre>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const tokenTemplates = {
            vision: '<|vision_start|><|image_pad|><|vision_end|>',
            spatial: '<|object_ref_start|>TARGET<|object_ref_end|> at <|box_start|>0,0,100,100<|box_end|>',
            chat: '<|im_start|>user\\nCUSTOM_TEXT<|im_end|>'
        };
        
        function generatePrompt() {
            const tokenType = document.getElementById('tokenType').value;
            const customText = document.getElementById('customText').value || 'Sample text';
            
            let template = tokenTemplates[tokenType];
            if (tokenType === 'spatial') {
                template = template.replace('TARGET', customText);
            } else if (tokenType === 'chat') {
                template = template.replace('CUSTOM_TEXT', customText);
            } else {
                template = customText + ' ' + template;
            }
            
            const promptDiv = document.getElementById('generatedPrompt');
            const preElement = promptDiv.querySelector('pre');
            preElement.textContent = template;
            promptDiv.classList.remove('hidden');
        }
    </script>
</body>
</html>
        '''
        
        return html_template
    
    def print_analysis(self):
        """Print comprehensive token analysis"""
        print("=== Qwen2.5-VL Tokenizer Analysis ===\n")
        
        # Basic stats
        print(f"Total added tokens: {len(self.added_tokens)}")
        print(f"Token ID range: {min(self.added_tokens.values())} - {max(self.added_tokens.values())}")
        print(f"Model max length: {self.config.get('model_max_length', 'Unknown')}\n")
        
        # Token categories
        categories = self.categorize_tokens()
        print("=== Token Categories ===")
        for category, tokens in categories.items():
            print(f"\n{category}:")
            for token, token_id in sorted(tokens, key=lambda x: x[1]):
                special_mark = " [SPECIAL]" if self.is_special_token(token) else ""
                print(f"  {token_id}: {token}{special_mark}")
        
        # Vision tokens analysis
        print("\n=== Vision Token Analysis ===")
        vision_tokens = self.analyze_vision_tokens()
        for token, info in vision_tokens.items():
            print(f"{token} (ID: {info['token_id']}):")
            print(f"  Purpose: {info['purpose']}")
            print(f"  Special: {info['is_special']}\n")
        
        # Spatial tokens analysis
        print("=== Spatial Reference Token Analysis ===")
        spatial_tokens = self.analyze_spatial_tokens()
        for token, info in spatial_tokens.items():
            print(f"{token} (ID: {info['token_id']}):")
            print(f"  Purpose: {info['purpose']}")
            print(f"  Special: {info['is_special']}\n")
        
        # Test prompts
        print("=== Generated Test Prompts ===")
        test_prompts = self.generate_test_prompts()
        for i, test in enumerate(test_prompts, 1):
            print(f"{i}. {test['name']}")
            print(f"   Purpose: {test['purpose']}")
            print(f"   Prompt: {test['prompt']}\n")


def main():
    """Main execution function"""
    analyzer = QwenTokenAnalyzer()
    analyzer.print_analysis()
    
    # Generate HTML interface
    html_content = analyzer.create_exploration_interface()
    html_path = "qwen_token_explorer.html"
    
    token_cards_html = ""
    categories = analyzer.categorize_tokens()
    for category, tokens in categories.items():
        token_list = "".join([f'<li class="text-sm">{token} (ID: {token_id})</li>' 
                             for token, token_id in sorted(tokens, key=lambda x: x[1])])
        
        token_cards_html += f'''
        <div class="bg-white rounded-lg shadow-md p-6">
            <h3 class="text-lg font-bold mb-3 text-gray-800">{category}</h3>
            <ul class="space-y-1 text-gray-600">
                {token_list}
            </ul>
        </div>
        '''
    
    html_content = html_content.replace('{token_cards}', token_cards_html)
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n=== Files Generated ===")
    print(f"HTML Token Explorer: {html_path}")
    print(f"Analysis Script: {__file__}")


if __name__ == "__main__":
    main()