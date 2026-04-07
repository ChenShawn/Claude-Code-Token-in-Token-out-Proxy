#!/bin/bash

# MODEL_PATH=/diancpfs/user/fengyuan/backbones/DeepSeek-V3.2
MODEL_PATH=/diancpfs/user/fengyuan/backbones/Qwen/Qwen3.5-35B-A3B
SGLANG_BASE_URL=http://10.146.228.161:18901
TOOL_PARSER=qwen3_coder

set -x
python proxy_serving.py \
    --sglang-base-url ${SGLANG_BASE_URL} \
    --tokenizer-path ${MODEL_PATH} \
    --tool-parser ${TOOL_PARSER} \
    --host 0.0.0.0 \
    --port 18901
