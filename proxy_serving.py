"""
1. 作为sglang的透明反向代理，接收client对sglang的请求（client使用OpenAI接口，例如`v1/completion`），转成token_ids输入给`sglang/generate`接口，拿到token ids返回以后伪装成sglang的返回，返回给client；
2. client是一个agent，会与sglang进行多轮交互，代码中把一条trajectory中所有的样本存下来，token_id存parquet文件，文本存json文件；
"""

import json
import logging
import os
import time
import uuid
import argparse
from urllib.parse import urlparse
from typing import List, Dict, Any, AsyncGenerator

import httpx
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from transformers import AutoTokenizer

from global_types import Message, OpenAICompletionRequest, OpenAIChatRequest
from tool_types import ToolParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# ===== 全局变量（由CLI注入） =====
SGLANG_URL = None
SGLANG_BASE_URL = None  # scheme://host:port，用于透传代理
PARQUET_PATH = None
JSON_PATH = None
TOKENIZER = None
TOOL_PARSER: ToolParser | None = None

# ===== 内存缓存 trajectory =====
trajectory_store: Dict[str, List[Dict[str, Any]]] = {}

# ===== tokenizer =====
def text_to_token_ids(text: str) -> List[int]:
    return TOKENIZER.encode(text, add_special_tokens=False)


def token_ids_to_text(token_ids: List[int], skip_special_tokens: bool = True) -> str:
    return TOKENIZER.decode(token_ids, skip_special_tokens=skip_special_tokens)


# ===== chat prompt 拼接 =====


def _message_content_to_str(content) -> str:
    """将 Message.content 统一转为字符串，支持 str / None / 多模态列表。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # 多模态内容列表：提取所有 text 部分，忽略 image 等
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def build_chat_prompt(messages: List[Message], tools: List[Dict[str, Any]] | None = None) -> str:
    # 使用 tokenizer 自带 chat template（推荐）
    chat = []
    for m in messages:
        msg = {"role": m.role, "content": _message_content_to_str(m.content)}
        if m.name is not None:
            msg["name"] = m.name
        if m.tool_call_id is not None:
            msg["tool_call_id"] = m.tool_call_id
        if m.tool_calls is not None:
            msg["tool_calls"] = m.tool_calls
        chat.append(msg)
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if tools:
        kwargs["tools"] = tools
    return TOKENIZER.apply_chat_template(chat, **kwargs)


# ===== sglang 调用 =====
async def call_sglang(
    token_ids: List[int],
    max_tokens: int,
    max_completion_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logit_bias: Dict[int, float] | None = None,
    stop: str | List[str] | None = None,
    extra_params: Dict[str, Any] | None = None,
):
    payload = {
        "input_ids": token_ids,
        "sampling_params": {
            "temperature": temperature,
        },
    }

    # max_completion_tokens 优先于 max_tokens（新版 API 兼容）
    if max_completion_tokens:
        payload["sampling_params"]["max_new_tokens"] = max_completion_tokens
    if max_tokens and not max_completion_tokens:
        payload["sampling_params"]["max_new_tokens"] = max_tokens

    # 添加可选的参数
    if top_p < 1.0:
        payload["sampling_params"]["top_p"] = top_p
    if presence_penalty > 0.0:
        payload["sampling_params"]["presence_penalty"] = presence_penalty
    if frequency_penalty > 0.0:
        payload["sampling_params"]["frequency_penalty"] = frequency_penalty
    if logit_bias is not None:
        payload["sampling_params"]["logit_bias"] = logit_bias

    # 处理stop参数，可以是字符串或字符串列表
    if stop is not None:
        if isinstance(stop, str):
            stop = [stop]
        payload["sampling_params"]["stop"] = stop

    # 透传额外参数到 sampling_params
    if extra_params:
        payload["sampling_params"].update(extra_params)

    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(SGLANG_URL, json=payload)
        resp.raise_for_status()
        return resp.json()


# ===== sglang streaming =====
# 注意：这里仍然依赖 sglang 返回 token_ids（不是 mock，但强依赖具体实现）
async def stream_sglang(
    token_ids: List[int],
    max_tokens: int,
    max_completion_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logit_bias: Dict[int, float] | None = None,
    stop: str | List[str] | None = None,
    extra_params: Dict[str, Any] | None = None,
) -> AsyncGenerator[List[int], None]:
    payload = {
        "input_ids": token_ids,
        "stream": True,
        "sampling_params": {
            "temperature": temperature,
        },
    }

    if top_p is not None and top_p < 1.0:
        payload["sampling_params"]["top_p"] = top_p
    if presence_penalty is not None and presence_penalty > 0.0:
        payload["sampling_params"]["presence_penalty"] = presence_penalty
    if frequency_penalty is not None and frequency_penalty > 0.0:
        payload["sampling_params"]["frequency_penalty"] = frequency_penalty

    # max_completion_tokens 优先于 max_tokens（新版 API 兼容）
    if max_completion_tokens:
        payload["sampling_params"]["max_new_tokens"] = max_completion_tokens
    if max_tokens and not max_completion_tokens:
        payload["sampling_params"]["max_new_tokens"] = max_tokens

    # 添加可选的参数
    if logit_bias is not None:
        payload["sampling_params"]["logit_bias"] = logit_bias

    if stop is not None:
        # 处理stop参数，可以是字符串或字符串列表
        if isinstance(stop, str):
            stop = [stop]
        payload["sampling_params"]["stop"] = stop

    # 透传额外参数
    if extra_params:
        payload.update(extra_params)

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", SGLANG_URL, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    pline = line.strip()
                    if line.startswith("data:"):
                        pline = pline[len("data:") :].strip()
                    if pline == "[DONE]":
                        break
                    data = json.loads(pline)
                    # ⚠️ 非标准：sglang 返回字段不统一，这里做兼容（不是 mock，但属于推测性适配）
                    ids = data.get("output_ids") or data.get("token_ids")
                    if ids:
                        yield ids
                except json.JSONDecodeError:
                    logger.warning("stream_sglang: failed to parse line: %r", line)
                    continue


# ===== 持久化 =====
def save_trajectory():
    rows = []
    for traj_id, steps in trajectory_store.items():
        for step_id, step in enumerate(steps):
            rows.append(
                {
                    "trajectory_id": traj_id,
                    "step": step_id,
                }
            )

    if rows:
        parquet_target_fp = os.path.join(PARQUET_PATH, "tokens.parquet")
        df = pd.DataFrame(rows)
        df.to_parquet(parquet_target_fp, index=False)

    json_target_fp = os.path.join(JSON_PATH, "text.json")
    with open(json_target_fp, "w", encoding="utf-8") as f:
        json.dump(trajectory_store, f, ensure_ascii=False, indent=2)
    logger.info("trajectory saved: %d trajectories", len(trajectory_store))


# ===== trajectory 管理 =====
def get_or_create_trajectory(request: Request) -> str:
    traj_id = request.headers.get("X-Trajectory-Id")

    if not traj_id:
        traj_id = str(uuid.uuid4())

    if traj_id not in trajectory_store:
        trajectory_store[traj_id] = []

    return traj_id


# ===== SSE 工具 =====
def sse_format(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    # SSE 规范要求：data 字段中的每一行都必须有独立的 "data: " 前缀
    lines = payload.split("\n")
    return "".join(f"data: {line}\n" for line in lines) + "\n"


# ===== /v1/completions =====
@app.post("/v1/completions")
async def proxy_completion(request: Request, body: OpenAICompletionRequest):
    # 检查不支持的高级参数并记录警告
    if body.n != 1:
        logger.warning(
            "OpenAICompletionRequest: n=%d is not fully supported (will only return 1 choice)",
            body.n,
        )
        body.n = 1

    if body.logprobs is not None:
        logger.warning("OpenAICompletionRequest: logprobs is not supported")
        body.logprobs = None

    if body.echo:
        logger.warning("OpenAICompletionRequest: echo is not supported")

    traj_id = get_or_create_trajectory(request)

    # prompt 可以是 str / List[str] / List[int]（token IDs）/ List[List[int]]
    prompt = body.prompt
    if isinstance(prompt, list):
        if len(prompt) == 0:
            prompt = ""
        elif isinstance(prompt[0], int):
            # List[int] —— 直接就是 token IDs
            input_ids = prompt
            prompt = token_ids_to_text(input_ids)
        elif isinstance(prompt[0], list):
            # List[List[int]] —— 取第一个（n>1 不支持）
            input_ids = prompt[0]
            prompt = token_ids_to_text(input_ids)
        else:
            # List[str] —— 取第一个
            prompt = prompt[0]

    if isinstance(prompt, str):
        input_ids = text_to_token_ids(prompt)
    # 如果上面的分支已经设置了 input_ids，这里 prompt 已被覆盖为 str

    if body.stream:
        cmpl_id = f"cmpl-{uuid.uuid4().hex}"

        async def generator():
            full_ids = []
            prev_text = ""
            async for ids in stream_sglang(
                input_ids,
                body.max_tokens,
                body.max_completion_tokens,
                body.temperature,
                body.top_p,
                body.presence_penalty,
                body.frequency_penalty,
                body.logit_bias,
                body.stop,
                extra_params=body.model_extra,
            ):
                full_ids = ids
                text = token_ids_to_text(ids)
                delta = text[len(prev_text) :]
                prev_text = text
                if delta:
                    yield sse_format(
                        {
                            "id": cmpl_id,
                            "object": "text_completion",
                            "choices": [
                                {"text": delta, "index": 0, "finish_reason": None}
                            ],
                        }
                    )

            # 最终 chunk 带 finish_reason
            yield sse_format(
                {
                    "id": cmpl_id,
                    "object": "text_completion",
                    "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                }
            )

            trajectory_store[traj_id].append(
                {
                    "input_text": prompt,
                    "output_text": prev_text,
                }
            )
            save_trajectory()

            yield "data: [DONE]\n\n"

        return StreamingResponse(generator(), media_type="text/event-stream")

    # 非 streaming
    sglang_resp = await call_sglang(
        input_ids,
        body.max_tokens,
        body.max_completion_tokens,
        body.temperature,
        body.top_p,
        body.presence_penalty,
        body.frequency_penalty,
        body.logit_bias,
        body.stop,
        extra_params=body.model_extra,
    )
    output_ids = sglang_resp.get("output_ids") or sglang_resp.get("token_ids", [])
    output_text = token_ids_to_text(output_ids)

    trajectory_store[traj_id].append(
        {
            "input_text": prompt,
            "output_text": output_text,
        }
    )
    save_trajectory()

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [{"text": output_text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(output_ids),
            "total_tokens": len(input_ids) + len(output_ids),
        },
    }


# ===== /v1/chat/completions =====
@app.post("/v1/chat/completions")
async def proxy_chat_completion(request: Request, body: OpenAIChatRequest):
    # 检查不支持的高级参数并记录警告
    if body.n != 1:
        logger.warning(
            "OpenAIChatRequest: n=%d is not fully supported (will only return 1 choice)",
            body.n,
        )
        body.n = 1

    if body.response_format is not None:
        logger.warning("OpenAIChatRequest: response_format is not supported")

    if body.functions is not None:
        logger.warning("OpenAIChatRequest: functions (legacy) is not supported")
    if body.function_call is not None:
        logger.warning("OpenAIChatRequest: function_call (legacy) is not supported")

    traj_id = get_or_create_trajectory(request)

    prompt = build_chat_prompt(body.messages, tools=body.tools)
    input_ids = text_to_token_ids(prompt)

    # Determine whether to preserve special tokens for tool parsing
    skip_sp = not (
        TOOL_PARSER
        and body.tools
        and getattr(TOOL_PARSER, "requires_no_skip_special_tokens", False)
    )
    use_tool_parser = TOOL_PARSER is not None and body.tools

    if body.stream:
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex}"

        async def generator():
            full_ids = []
            prev_text = ""
            prev_ids: List[int] = []
            has_tool_calls = False
            async for ids in stream_sglang(
                input_ids,
                body.max_tokens,
                body.max_completion_tokens,
                body.temperature,
                body.top_p,
                body.presence_penalty,
                body.frequency_penalty,
                body.logit_bias,
                body.stop,
                extra_params=body.model_extra,
            ):
                full_ids = ids
                text = token_ids_to_text(ids, skip_special_tokens=skip_sp)
                delta = text[len(prev_text) :]
                delta_ids = ids[len(prev_ids):]

                if use_tool_parser and TOOL_PARSER:
                    msg = TOOL_PARSER.extract_tool_calls_streaming(
                        previous_text=prev_text,
                        current_text=text,
                        delta_text=delta,
                        previous_token_ids=prev_ids,
                        current_token_ids=ids,
                        delta_token_ids=delta_ids,
                        request=body,
                    )
                    prev_text = text
                    prev_ids = list(ids)

                    if msg is None:
                        continue

                    chunk_delta: Dict[str, Any] = {}
                    if msg.content:
                        chunk_delta["content"] = msg.content
                    if msg.tool_calls:
                        has_tool_calls = True
                        chunk_delta["tool_calls"] = [
                            tc.model_dump(exclude_none=True) for tc in msg.tool_calls
                        ]

                    if chunk_delta:
                        yield sse_format(
                            {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {
                                        "delta": chunk_delta,
                                        "index": 0,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                else:
                    prev_text = text
                    prev_ids = list(ids)
                    if delta:
                        yield sse_format(
                            {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {
                                        "delta": {"content": delta},
                                        "index": 0,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )

            # Final chunk with finish_reason
            finish_reason = "tool_calls" if has_tool_calls else "stop"
            yield sse_format(
                {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "delta": {},
                            "index": 0,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            )

            trajectory_store[traj_id].append(
                {
                    "input_text": prompt,
                    "output_text": prev_text,
                    "messages": [m.model_dump() for m in body.messages],
                }
            )
            save_trajectory()

            yield "data: [DONE]\n\n"

        return StreamingResponse(generator(), media_type="text/event-stream")

    # 非 streaming
    # Determine whether to preserve special tokens for tool parsing
    skip_sp = not (
        TOOL_PARSER
        and body.tools
        and getattr(TOOL_PARSER, "requires_no_skip_special_tokens", False)
    )

    sglang_resp = await call_sglang(
        input_ids,
        body.max_tokens,
        body.max_completion_tokens,
        body.temperature,
        body.top_p,
        body.presence_penalty,
        body.frequency_penalty,
        body.logit_bias,
        body.stop,
        extra_params=body.model_extra,
    )
    output_ids = sglang_resp.get("output_ids") or sglang_resp.get("token_ids", [])
    output_text = token_ids_to_text(output_ids, skip_special_tokens=skip_sp)

    trajectory_store[traj_id].append(
        {
            "input_text": prompt,
            "output_text": output_text,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "messages": [m.model_dump() for m in body.messages],
        }
    )
    save_trajectory()

    # Try to extract tool calls from model output
    message: Dict[str, Any] = {"role": "assistant", "content": output_text}
    finish_reason = "stop"

    if TOOL_PARSER and body.tools:
        extracted = TOOL_PARSER.extract_tool_calls(output_text, body)
        if extracted.tools_called and extracted.tool_calls:
            finish_reason = "tool_calls"
            message["content"] = extracted.content
            message["tool_calls"] = [
                tc.model_dump() for tc in extracted.tool_calls
            ]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(output_ids),
            "total_tokens": len(input_ids) + len(output_ids),
        },
    }


# ===== 透传其他所有请求 =====


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_all(request: Request, path: str):
    # 跳过已经实现的接口
    if path.startswith("v1/completions") or path.startswith("v1/chat/completions"):
        return {"error": "handled by specific route"}

    url = f"{SGLANG_BASE_URL}/{path}"

    headers = dict(request.headers)
    headers.pop("host", None)

    body = await request.body()

    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.request(
            method=request.method, url=url, headers=headers, content=body
        )

    # 透传 status code、Content-Type 和响应 body
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


# ===== CLI 启动 =====


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-base-url", type=str, required=True)
    parser.add_argument("--parquet-path", type=str, default="./data")
    parser.add_argument("--json-path", type=str, default="./data")
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tool-parser",
        type=str,
        default=None,
        choices=["qwen3_coder", "deepseek_v32"],
        help="Tool call parser to use for extracting tool calls from model output",
    )
    return parser.parse_args()


def init_globals(args):
    global SGLANG_URL, SGLANG_BASE_URL, PARQUET_PATH, JSON_PATH, TOKENIZER, TOOL_PARSER
    SGLANG_BASE_URL = args.sglang_base_url
    SGLANG_URL = f"{SGLANG_BASE_URL}/generate"
    PARQUET_PATH = args.parquet_path
    JSON_PATH = args.json_path

    # 确保数据目录存在
    for path in [PARQUET_PATH, JSON_PATH]:
        dirname = os.path.dirname(os.path.abspath(path))
        os.makedirs(dirname, exist_ok=True)

    logger.info("Loading tokenizer from %s ...", args.tokenizer_path)
    TOKENIZER = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    logger.info("SGLang URL: %s, base: %s", SGLANG_URL, SGLANG_BASE_URL)

    # Initialize tool parser
    tool_parser_name = getattr(args, "tool_parser", None)
    if tool_parser_name == "qwen3_coder":
        from qwen3coder_tool_parser import Qwen3CoderToolParser
        TOOL_PARSER = Qwen3CoderToolParser(TOKENIZER)
        logger.info("Tool parser: Qwen3CoderToolParser")
    elif tool_parser_name == "deepseek_v32":
        from deepseekv32_tool_parser import DeepSeekV32ToolParser
        TOOL_PARSER = DeepSeekV32ToolParser(TOKENIZER)
        logger.info("Tool parser: DeepSeekV32ToolParser")
    else:
        TOOL_PARSER = None


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    init_globals(args)
    uvicorn.run(app, host=args.host, port=args.port)
