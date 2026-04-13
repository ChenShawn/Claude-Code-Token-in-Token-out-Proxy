set -x

# docker run -it --name test-proxy --entrypoint /bin/bash -p 18080:18080 -v /diancpfs:/diancpfs claude-code-token-in-token-out-proxy:latest

docker run -d --name test-proxy -p 18080:18080 -v /diancpfs:/diancpfs chenshawn6915/claude-code-token-in-token-out-proxy:latest
