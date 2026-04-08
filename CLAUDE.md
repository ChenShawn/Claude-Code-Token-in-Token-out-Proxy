# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A transparent reverse proxy that sits between OpenAI API clients and an SGLang inference server. It:
1. Receives OpenAI-compatible requests (`/v1/completions`, `/v1/chat/completions`)
2. Converts text to token IDs using a HuggingFace tokenizer, forwards to SGLang's `/generate` endpoint
3. Converts SGLang's token ID responses back to OpenAI-format responses
4. Records all request/response pairs (trajectories) grouped by `X-Trajectory-Id` header, saving to parquet and JSON files
5. Transparently proxies all other routes to the upstream SGLang server

## Running the Server

```bash
python proxy_serving.py \
    --sglang-base-url <SGLANG_URL> \
    --tokenizer-path <MODEL_PATH> \
    --host 0.0.0.0 \
    --port 18901
```

See `run_serving.sh` for a working example with real model paths.

## Architecture

- **`proxy_serving.py`** — Single-file FastAPI app. All routing, SGLang communication (both streaming and non-streaming), trajectory storage, and the CLI entrypoint live here. Global state (`SGLANG_URL`, `TOKENIZER`, `trajectory_store`) is initialized via `init_globals()` from CLI args.
- **`global_types.py`** — Pydantic models for OpenAI API request/response types (`Message`, `OpenAICompletionRequest`, `OpenAIChatRequest`) plus trajectory storage classes (`AgentTrajectory`, `TrajectoryStore`). Models use `extra = "allow"` to pass through unknown fields.
- **`tool_parsers/`** — Tool call parsing package. Contains:
  - `tool_types.py` — Base data types (`ToolCall`, `DeltaToolCall`, `ExtractedToolCallInformation`, etc.) and abstract `ToolParser` class.
  - `deepseekv32_tool_parser.py` — DeepSeek V3.2 DSML format parser.
  - `qwen3coder_tool_parser.py` — Qwen3 Coder XML format parser.
- **`tests/`** — Test files for tool parsers (`test_deepseekv32_tool_parser.py`, `test_qwen3coder_tool_parser.py`).
- **`data/`** — Output directory for trajectory data (`tokens.parquet`, `text.json`).

## Key Design Decisions

- Chat messages are converted to a single prompt string using the tokenizer's `apply_chat_template`, then tokenized — the proxy operates at the token ID level with SGLang.
- `max_completion_tokens` takes precedence over `max_tokens` (newer OpenAI API convention).
- Streaming uses SSE (Server-Sent Events) with incremental text deltas computed by diffing decoded output against previous text.
- Trajectory persistence happens synchronously after every request (both streaming and non-streaming).
- The catch-all route (`/{path:path}`) forwards unrecognized requests directly to the SGLang base URL.

## Dependencies

Python packages: `fastapi`, `uvicorn`, `httpx`, `pandas`, `transformers`, `pydantic`
