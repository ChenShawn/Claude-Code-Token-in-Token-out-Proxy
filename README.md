# Relevant Documentation
- [使用文档](./docs/usage.md)
- [agent存储格式设计文档](./docs/agent_format.md)

# Token-In-Token-Out Claude-Code Proxy Serving

**The missing piece for Agentic RL training: a token-in, token-out proxy for SGLang.**

The retokenization issueL: why does it matter?

The mismatch between the inference engine
Most OpenAI-compatible proxies work at the **text level** -- they decode tokens to strings, then re-encode on the next turn. This introduces retokenization drift: the token IDs your model sees during training don't match what it saw during rollout. For supervised fine-tuning this is a minor nuisance. **For RL training, it's a silent killer of reward signal.**

```
Claude-Code → LiteLLM  →  [Proxy: FastAPI]  →  vLLM/SGLang
                                 ↓
                          Tokenizer (Local)
                                 ↓
                          TrajectoryDumper
```

## Key Features

- **Token-level fidelity** -- prompt and response token IDs are tracked across multi-turn conversations with zero tokenization drift. Previous-turn tokens are reused as prefix, not re-encoded.
- **Drop-in OpenAI compatibility** -- supports `/v1/chat/completions` and `/v1/completions` with streaming, tool calling, and all standard parameters. Point any OpenAI client at the proxy and it just works.
- **Automatic trajectory recording** -- every agent conversation is saved with token IDs + response masks (parquet) and full message history (JSONL), structured for RL/SFT training pipelines.
- **Tool call parsing** -- built-in parsers for DeepSeek-V3.2 (DSML) and Qwen3-Coder (XML) extract structured tool calls from raw model output, with streaming delta support.
- **Transparent passthrough** -- any route the proxy doesn't handle is forwarded directly to SGLang, so `/v1/models`, health checks, etc. all work unchanged.

## Quick Start

### 1. Install dependencies

```bash
pip install fastapi uvicorn httpx pandas transformers pydantic
```

### 2. Start the proxy

```bash
python proxy_serving.py \
    --sglang-base-url http://localhost:30000 \
    --tokenizer-path /path/to/your/model \
    --tool-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 18901
```

### 3. Point your client at the proxy

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:18901/v1",
    api_key="unused",  # no auth required
)

response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

That's it. The proxy handles tokenization, format translation, and trajectory recording transparently.

## CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--sglang-base-url` | Yes | -- | URL of the upstream SGLang server (e.g. `http://localhost:30000`) |
| `--tokenizer-path` | Yes | -- | Path to a HuggingFace model or tokenizer directory |
| `--tool-parser` | No | `None` | Tool call parser: `qwen3_coder` or `deepseek_v32` |
| `--host` | No | `0.0.0.0` | Bind address |
| `--port` | No | `8000` | Bind port |
| `--parquet-path` | No | `./data` | Output directory for parquet trajectory files |
| `--json-path` | No | `./data` | Output directory for JSONL trajectory files |

See [`run_serving.sh`](run_serving.sh) for a ready-to-use example.

## Supported Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat completions (streaming and non-streaming) |
| `POST /v1/completions` | Text completions (streaming and non-streaming) |
| `* /{path}` | All other routes are transparently proxied to SGLang |

Both endpoints support the full range of OpenAI parameters: `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `stop`, `frequency_penalty`, `presence_penalty`, `logit_bias`, `tools`, `stream`, etc.

## How It Works

```
OpenAI Client                Proxy                        SGLang
     |                         |                             |
     |-- POST /v1/chat/completions -->                       |
     |                         |                             |
     |              apply_chat_template()                     |
     |              tokenize -> input_ids                     |
     |                         |                             |
     |                         |-- POST /generate ---------->|
     |                         |   (token IDs in/out)        |
     |                         |<--- output token IDs -------|
     |                         |                             |
     |              decode token IDs -> text                  |
     |              extract tool calls (if parser set)        |
     |              save trajectory to disk                   |
     |                         |                             |
     |<-- OpenAI-format JSON --|                             |
```

1. Client sends a standard OpenAI API request
2. Messages are converted to a prompt string via the tokenizer's `apply_chat_template`, then tokenized to IDs
3. Token IDs are sent to SGLang's `/generate` endpoint
4. Response token IDs are decoded back to text, wrapped in OpenAI-format JSON
5. If a tool parser is configured, tool calls are extracted from the model output
6. The full request/response pair is persisted to `data/` (parquet for token IDs, JSONL for text)

For multi-turn agent conversations, the proxy reuses token IDs from previous turns to avoid re-encoding, keeping prompts consistent across turns.

## Trajectory Recording

Every request is automatically recorded. Trajectories are grouped by agent (matched via conversation prefix) and saved in two formats:

- **Parquet** (`data/<traj_id>.parquet`) -- prompt token IDs, response token IDs, and response masks per agent. Suitable for training pipelines.
- **JSONL** (`data/<traj_id>.jsonl`) -- full message history per agent in OpenAI format. Suitable for inspection and replay.

Each agent's trajectory includes metadata: number of turns, prompt/response/observation token counts, and timestamps.

## Tool Call Parsing

When `--tool-parser` is set and the request includes `tools`, the proxy extracts structured tool calls from the model's raw text output:

| Parser | Model | Format |
|---|---|---|
| `qwen3_coder` | Qwen3-Coder family | XML-based tool calls |
| `deepseek_v32` | DeepSeek-V3.2 | DSML format |

Tool calls are returned in the standard OpenAI `tool_calls` format, with `finish_reason: "tool_calls"`. Streaming tool call deltas are also supported.

## Project Structure

```
.
├── proxy_serving.py                  # FastAPI app: routing, SGLang communication, trajectory I/O
├── global_types.py                   # Pydantic models (OpenAI request/response types, trajectory storage)
├── tool_parsers/
│   ├── tool_types.py                 # Base ToolParser interface and data types
│   ├── qwen3coder_tool_parser.py     # Qwen3-Coder XML tool call parser
│   ├── deepseekv32_tool_parser.py    # DeepSeek-V3.2 DSML tool call parser
│   └── reasoning_parser.py           # Reasoning content extraction (think blocks)
├── tests/                            # Unit tests and demo scripts
├── run_serving.sh                    # Example launch script
├── Makefile                          # clean / install / test targets
└── data/                             # Output directory for trajectory files
```

## Development

```bash
# Run tests
make test

# Clean generated files
make clean
```
