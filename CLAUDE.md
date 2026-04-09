# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A transparent reverse proxy that sits between OpenAI API clients and an SGLang inference server. It:
1. Receives OpenAI-compatible requests (`/v1/completions`, `/v1/chat/completions`)
2. Converts text to token IDs using a HuggingFace tokenizer, forwards to SGLang's `/generate` endpoint
3. Converts SGLang's token ID responses back to OpenAI-format responses, including tool call parsing and reasoning extraction
4. Records all request/response pairs (trajectories) grouped by agent, saving token IDs to parquet and conversation text to JSONL
5. Transparently proxies all other routes to the upstream SGLang server

Optionally runs behind a LiteLLM gateway (see `config/config.yaml`) that maps model names like `claude-*` to the proxy's OpenAI-compatible endpoint.

## Common Commands

```bash
make install       # pip install -r requirements.txt
make test          # python -m pytest tests/ -v
make clean         # rm data/, log/, __pycache__, .pytest_cache

# Run a single test file
python -m pytest tests/test_deepseekv32_tool_parser.py -v

# Run a single test class or method
python -m pytest tests/test_deepseekv32_tool_parser.py::TestExtractToolCalls::test_single_tool_call_one_param -v
```

## Running the Server

```bash
python proxy_serving.py \
    --sglang-base-url <SGLANG_URL> \
    --tokenizer-path <MODEL_PATH> \
    --tool-parser qwen3_coder \  # or deepseek_v32
    --host 0.0.0.0 \
    --port 18901
```

See `run_serving.sh` for a working example. It also starts a LiteLLM proxy on port 18080.

## Architecture

- **`proxy_serving.py`** — Single-file FastAPI app. All routing, SGLang communication (both streaming and non-streaming), trajectory storage, and the CLI entrypoint live here. Global state (`SGLANG_URL`, `TOKENIZER`, `TOOL_PARSER`, `trajectory_store`) is initialized via `init_globals()` from CLI args.
- **`global_types.py`** — Pydantic models for OpenAI API request/response types (`Message`, `OpenAICompletionRequest`, `OpenAIChatRequest`) plus trajectory storage classes (`AgentTrajectory`, `TrajectoryStore`). Models use `extra = "allow"` to pass through unknown fields from clients.
- **`tool_parsers/`** — Tool call parsing package:
  - `tool_types.py` — Base data types (`ToolCall`, `DeltaToolCall`, `ExtractedToolCallInformation`, etc.) and abstract `ToolParser` class with reasoning integration.
  - `reasoning_parser.py` — Extracts `<think>...</think>` reasoning blocks from model output. Supports both streaming (stateful) and non-streaming (one-shot) modes.
  - `deepseekv32_tool_parser.py` — DeepSeek V3.2 DSML format parser (fullwidth pipe `U+FF5C` delimiters).
  - `qwen3coder_tool_parser.py` — Qwen3 Coder XML format parser (`<tool_call>/<function=...>/<parameter=...>`).
  - `glm4_moe_tool_parser.py`, `glm47_moe_tool_parser.py` — GLM model tool parsers.
- **`tests/`** — Tests use mock tokenizers (no HuggingFace downloads needed). Test both non-streaming `extract_tool_calls()` and streaming `extract_tool_calls_streaming()`.
- **`vllm/`** — Vendored reference implementations from vLLM (gitignored, not part of the project source). Used as reference when porting tool parsers.

## Key Design Decisions

- Chat messages are converted to a single prompt string using the tokenizer's `apply_chat_template`, then tokenized — the proxy operates at the token ID level with SGLang.
- `max_completion_tokens` takes precedence over `max_tokens` (newer OpenAI API convention).
- Streaming uses SSE (Server-Sent Events) with incremental text deltas. Reasoning deltas (`reasoning_content`) and content deltas are emitted separately.
- Tool parsers are per-request copies (`copy.copy`) to avoid concurrent state corruption. Each copy gets `_reset_streaming_state()` and `init_reasoning(prompt)` called before use.
- Trajectory storage tracks multi-turn agent conversations. `AgentTrajectory.matches_prefix()` identifies returning agents by comparing normalized message history. Token IDs are split into `prompt_token_ids`, `response_token_ids`, and `response_mask` (1 for tool/observation tokens, 0 for model output).
- The catch-all route (`/{path:path}`) forwards unrecognized requests directly to the SGLang base URL.
- Tool parsers convert parameter values to typed Python objects (int, bool, float, etc.) using the tool schema before serializing to JSON arguments.

## Adding a New Tool Parser

1. Create `tool_parsers/<model>_tool_parser.py` subclassing `ToolParser` from `tool_types.py`
2. Implement `extract_tool_calls()` (non-streaming) and `extract_tool_calls_streaming()` (streaming)
3. If the model uses `<think>` reasoning, set `supports_reasoning = True` and override `init_reasoning()`
4. If the model requires special tokens in decoded output, set `requires_no_skip_special_tokens = True`
5. Register in `tool_parsers/__init__.py` and add the CLI choice in `proxy_serving.py:parse_args()`
6. Add tests with a mock tokenizer in `tests/`
