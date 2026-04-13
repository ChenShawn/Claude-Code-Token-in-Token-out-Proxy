set -x

# docker run -it --name test-proxy --entrypoint /bin/bash -p 18080:18080 -v /diancpfs:/diancpfs claude-code-token-in-token-out-proxy:latest

docker run -d \
    -p 18080:18080 \
    -v /diancpfs:/diancpfs \
    -e SGLANG_TARGET_URL=http://10.146.236.83:30000 \
    -e LOCAL_MODEL_PATH=/diancpfs/user/fengyuan/backbones/GLM-5.1 \
    -e TOOL_PARSER_NAME=glm47 \
    chenshawn6915/claude-code-token-in-token-out-proxy:latest
