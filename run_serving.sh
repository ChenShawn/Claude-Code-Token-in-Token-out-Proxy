#!/bin/bash

# MODEL_PATH=/diancpfs/user/fengyuan/backbones/DeepSeek-V3.2
# MODEL_PATH=/diancpfs/user/fengyuan/backbones/Qwen/Qwen3.5-35B-A3B
MODEL_PATH=/diancpfs/user/fengyuan/backbones/GLM-5.1
# SGLANG_BASE_URL=http://10.146.225.70:18901
SGLANG_BASE_URL=http://10.146.236.83:30000

# TOOL_PARSER=qwen3_coder
# TOOL_PARSER=deepseek_v32
TOOL_PARSER=glm47

make clean
mkdir -p ./log
mkdir -p ./data

set -x

litellm --config ./config/config.yaml --port 18080 2>&1 &

python proxy_serving.py \
    --sglang-base-url ${SGLANG_BASE_URL} \
    --tokenizer-path ${MODEL_PATH} \
    --tool-parser ${TOOL_PARSER} \
    --host 0.0.0.0 \
    --port 18901 2>&1 | tee ./log/proxy.log
