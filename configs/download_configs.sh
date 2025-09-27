#!/bin/bash

# Download Qwen2-VL-7B-Instruct config files
echo "Downloading Qwen2.5-VL config files..."

cd qwen25vl/

# Main config files
echo "Downloading main configs..."
curl -L -O https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/config.json
curl -L -O https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/preprocessor_config.json

# Vision config
echo "Downloading vision config..."
curl -L -O https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/vision_config.json

# Tokenizer files
echo "Downloading tokenizer files..."
curl -L -O https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/tokenizer.json
curl -L -O https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/tokenizer_config.json
curl -L -O https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/special_tokens_map.json
curl -L -O https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/vocab.json
curl -L -O https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/merges.txt

# Chat template
echo "Downloading chat template..."
curl -L -O https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/chat_template.json

# Generation config
echo "Downloading generation config..."
curl -L -O https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/generation_config.json

# Model index (for safetensors)
echo "Downloading model index..."
curl -L -O https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/raw/main/model.safetensors.index.json

echo "Done! All config files downloaded to configs/qwen25vl/"
ls -la
