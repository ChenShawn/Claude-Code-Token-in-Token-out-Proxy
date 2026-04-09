# 使用文档

本项目是一个透明反向代理，接收 OpenAI 兼容的 API 请求，转换为 token ID 级别的请求发送给 SGLang 推理服务器，并将响应转换回 OpenAI 格式返回给客户端。同时记录所有请求/响应的轨迹数据（token ID 存 parquet，文本存 JSONL）。

## 环境要求

- Python >= 3.10
- 已部署并运行的 SGLang 推理服务器
- HuggingFace 格式的模型 tokenizer（本地路径）

### 前置依赖

```bash
make install
```

**非常重要：** Claude-Code **必须修改** 本地配置 `~/.claude/settings.json`，增加`env`中的环境变量：
```json
{
  "permissions": {
    "allow": [
      "Bash(python3 -c \"import json,sys; d=json.load\\(sys.stdin\\); print\\(list\\(d.keys\\(\\)\\)\\)\")",
      "Bash(python3 -c \"import json,sys; d=json.load\\(sys.stdin\\); print\\(json.dumps\\(d, indent=2\\)\\)\")",
      "Bash(ls:*)"
    ],
    "defaultMode": "acceptEdits"
  },
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ATTRIBUTION_HEADER" : "0"
  }
}
```

> 缺少这一步会导致性能下降90%，agent存储下来的拓扑结构不正确；

## 快速开始

最小化启动命令：

```bash
bash run_serving.sh
```

本地Claude-Code配置：
```bash
export ANTHROPIC_AUTH_TOKEN=123
export ANTHROPIC_BASE_URL=http://127.0.0.1:18080
```

## 配置参数详解

`proxy_serving.py` 支持以下命令行参数：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--sglang-base-url` | 是 | - | SGLang 服务器地址，如 `http://10.146.225.70:18901` |
| `--tokenizer-path` | 是 | - | HuggingFace tokenizer 路径，需与 SGLang 加载的模型一致 |
| `--tool-parser` | 否 | `None` | Tool calling 解析器，可选 `qwen3_coder` 或 `deepseek_v32` |
| `--host` | 否 | `0.0.0.0` | 代理服务监听地址 |
| `--port` | 否 | `8000` | 代理服务监听端口 |
| `--parquet-path` | 否 | `./data` | parquet 轨迹数据输出目录 |
| `--json-path` | 否 | `./data` | JSONL 轨迹数据输出目录 |

## LiteLLM 网关配置

项目支持在代理前面加一层 LiteLLM 网关，用于将任意模型名称（如 `claude-*`）映射到本代理的 OpenAI 接口。配置文件位于 `config/config.yaml`：

```yaml
model_list:
  - model_name: claude-*
    litellm_params:
      model: hosted_vllm/qwen35
      api_base: http://127.0.0.1:18901/v1
      api_key: ""
    model_info:
      max_tokens: 163840
      max_input_tokens: 131072
      max_output_tokens: 8192

litellm_settings:
  drop_params: true      # 丢弃 LiteLLM 不支持的参数，避免报错
  modify_params: true

general_settings:
  disable_key_check: true  # 不校验 API Key
```

启动 LiteLLM 网关：

```bash
litellm --config ./config/config.yaml --port 18080
```

启动后，客户端可以向 `http://<HOST>:18080` 发送请求，使用 `claude-sonnet-4-20250514` 等模型名称，LiteLLM 会自动路由到本代理。

`run_serving.sh` 中已包含同时启动代理和 LiteLLM 的完整流程。

## 数据输出

每次代理服务启动会生成一个 trajectory ID，所有轨迹数据保存在 `--parquet-path` 和 `--json-path` 指定的目录下（默认 `./data`）。

### JSONL 文件（`<traj_id>.jsonl`）

每行一个 agent 的完整对话记录：

```json
{
  "agent_id": "uuid",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."},
    ...
  ],
  "tools": [...],
  "metadata": {
    "num_turns": 3,
    "total_prompt_tokens": 1024,
    "total_obs_tokens": 256,
    "total_resp_tokens": 512,
    "total_agent_tokens": 768,
    "create_time": "2026-04-09T10:30:00",
    "update_time": "2026-04-09T10:35:00"
  }
}
```

- `total_obs_tokens`：observation token 数量（tool 返回结果对应的 token，`response_mask=1` 的部分）
- `total_resp_tokens`：模型实际生成的 token 数量（`response_mask=0` 的部分）
- `total_agent_tokens`：`response_token_ids` 的总长度（obs + resp）

### Parquet 文件（`<traj_id>.parquet`）

每行一个 agent，包含 token ID 级别的数据：

| 字段 | 说明 |
|------|------|
| `agent_id` | Agent 唯一标识 |
| `prompt_token_ids` | 输入 prompt 的 token ID 列表 |
| `response_token_ids` | 模型输出的 token ID 列表（包含 observation token） |
| `response_mask` | 与 `response_token_ids` 等长的 mask，1 表示 tool observation token，0 表示模型生成 token |
| `prompt_text` | prompt 的解码文本（含特殊 token） |
| `response_text` | response 的解码文本（含特殊 token） |

## 轨迹可视化工具

使用 `scripts/visualize_trajectory.py` 查看轨迹数据：

```bash
# 交互式浏览（默认模式，终端直接运行时）
python scripts/visualize_trajectory.py data/<traj_id>.jsonl

# 查看摘要表格
python scripts/visualize_trajectory.py data/<traj_id>.jsonl --summary

# 查看对话流（仅 user/assistant 文本）
python scripts/visualize_trajectory.py data/<traj_id>.jsonl --flow

# 查看指定 turn 的详情
python scripts/visualize_trajectory.py data/<traj_id>.jsonl --turn 3

# 查看指定 turn 的完整内容（不截断）
python scripts/visualize_trajectory.py data/<traj_id>.jsonl --turn 3 --full
```

交互模式下支持的命令：
- 输入数字 `N` 查看第 N 个 turn
- `f N` 查看第 N 个 turn 的完整内容
- `s` 显示摘要表格
- `c` 显示对话流
- `q` 退出

## Tool Calling 支持

通过 `--tool-parser` 参数启用 tool calling 解析，目前支持：

| 解析器 | 参数值 | 适用模型 |
|--------|--------|----------|
| Qwen3 Coder | `qwen3_coder` | Qwen3 Coder / Qwen 3.5 系列 |
| DeepSeek V3.2 | `deepseek_v32` | DeepSeek V3.2 |

启用后，代理会自动从模型输出中提取 tool call，并在 OpenAI 格式的响应中返回结构化的 `tool_calls` 字段。同时支持 streaming 和非 streaming 模式。

不指定 `--tool-parser` 时，代理仅做纯文本转发，不解析 tool call。

## 常用运维命令

```bash
# 清理数据和缓存
make clean

# 运行测试
make test

# 查看代理日志（run_serving.sh 启动时）
tail -f ./log/proxy.log

# 查看数据输出目录
ls -la ./data/
```

## 路由说明

| 路径 | 说明 |
|------|------|
| `POST /v1/completions` | 文本补全接口，支持 streaming |
| `POST /v1/chat/completions` | 聊天补全接口，支持 streaming、tool calling、reasoning |
| `* /{path}` | 其他所有请求透传到 SGLang 服务器 |
