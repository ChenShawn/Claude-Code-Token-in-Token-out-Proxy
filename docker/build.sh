set -x

docker build --no-cache -t claude-code-token-in-token-out-proxy:latest .
docker tag claude-code-token-in-token-out-proxy:latest chenshawn6915/claude-code-token-in-token-out-proxy:latest
docker push chenshawn6915/claude-code-token-in-token-out-proxy:latest

