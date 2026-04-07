# Claude Proxy GPT

A transparent reverse proxy that sits between OpenAI API clients and an SGLang/vLLM inference server. It enables trajectory recording for agent interactions while maintaining full OpenAI API compatibility.

## Features

- **OpenAI API Compatible**: Accepts standard `/v1/completions` and `/v1/chat/completions` requests
- **Token-level Proxying**: Converts text to token IDs via HuggingFace tokenizer, forwards to SGLang's `/generate` endpoint, and converts responses back to OpenAI format
- **Streaming Support**: Full SSE streaming with incremental text deltas
- **Trajectory Recording**: Records all request/response pairs grouped by `X-Trajectory-Id` header, saving to parquet (token IDs) and JSON (text) files
- **Tool Call Parsing**: Supports tool call parsing for DeepSeek-V3.2 and Qwen3-Coder models
- **Transparent Proxy**: Unrecognized routes are forwarded directly to the upstream server

## Quick Start

### Prerequisites

```bash
pip install fastapi uvicorn httpx pandas transformers pydantic
```

### Running

```bash
python proxy_serving.py \
    --sglang-base-url http://<SGLANG_HOST>:<PORT> \
    --tokenizer-path <MODEL_PATH> \
    --tool-parser <TOOL_PARSER> \
    --host 0.0.0.0 \
    --port 18901
```

See `run_serving.sh` for a working example.

### Arguments

| Argument | Description |
|---|---|
| `--sglang-base-url` | URL of the upstream SGLang server |
| `--tokenizer-path` | Path to the HuggingFace model/tokenizer |
| `--tool-parser` | Tool call parser to use (`deepseek_v32`, `qwen3_coder`) |
| `--host` | Bind host (default: `0.0.0.0`) |
| `--port` | Bind port (default: `18901`) |

## Project Structure

```
.
├── proxy_serving.py              # Main FastAPI app (routing, SGLang communication, trajectory storage)
├── global_types.py               # Pydantic models for OpenAI API types and trajectory storage
├── tool_types.py                 # Tool call parser interface
├── deepseekv32_tool_parser.py    # DeepSeek-V3.2 tool call parser
├── qwen3coder_tool_parser.py     # Qwen3-Coder tool call parser
├── run_serving.sh                # Example launch script
├── data/                         # Output directory for trajectory data
└── test_*.py                     # Unit tests for tool parsers
```

## How It Works

1. Client sends an OpenAI-compatible request with an `X-Trajectory-Id` header
2. The proxy applies `chat_template` to convert messages into a prompt string, then tokenizes it
3. Token IDs are sent to SGLang's `/generate` endpoint
4. SGLang's token ID response is decoded and wrapped in OpenAI-format response
5. The request/response pair is recorded to `data/` (parquet for tokens, JSON for text)
