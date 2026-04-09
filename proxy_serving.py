"""
1. 作为sglang的透明反向代理，接收client对sglang的请求（client使用OpenAI接口，例如`v1/completion`），转成token_ids输入给`sglang/generate`接口，拿到token ids返回以后伪装成sglang的返回，返回给client；
2. client是一个agent，会与sglang进行多轮交互，代码中把一条trajectory中所有的样本存下来，token_id存parquet文件，文本存json文件；
"""

import copy
import json
import logging
import os
import time
import uuid
import argparse
from urllib.parse import urlparse
from typing import List, Dict, Any, AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from transformers import AutoTokenizer

from global_types import Message, OpenAICompletionRequest, OpenAIChatRequest, TrajectoryStore
from tool_parsers.tool_types import ToolParser

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

# ===== 单例 trajectory store =====
trajectory_store: TrajectoryStore = None

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


def _to_plain(obj):
    """Recursively convert to plain Python dicts/lists (strip Pydantic models etc.)."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, 'model_dump'):
        return _to_plain(obj.model_dump())
    return obj


def _normalize_tool_calls(tool_calls):
    """Ensure tool_calls[].function.arguments is a dict, not a JSON string.

    The OpenAI API returns arguments as a JSON string, but chat templates
    (e.g. Qwen) expect a dict so they can call .items() on it.
    """
    result = _to_plain(tool_calls)
    for tc in result:
        func = tc.get("function", {})
        args = func.get("arguments")
        if isinstance(args, str):
            try:
                func["arguments"] = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass
    return result


def build_chat_prompt(messages: List[Message], tools: List[Dict[str, Any]] | None = None, add_generation_prompt: bool = True) -> str:
    # 使用 tokenizer 自带 chat template（推荐）
    chat = []
    for m in messages:
        if isinstance(m, Message):
            msg = {"role": m.role, "content": _message_content_to_str(m.content)}
            if m.reasoning_content is not None:
                msg["reasoning_content"] = m.reasoning_content
            if m.name is not None:
                msg["name"] = m.name
            if m.tool_call_id is not None:
                msg["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                msg["tool_calls"] = _normalize_tool_calls(m.tool_calls)
            chat.append(msg)
        else:
            msg = {"role": m["role"], "content": _message_content_to_str(m["content"])}
            if m.get("reasoning_content", None) is not None:
                msg["reasoning_content"] = m.get("reasoning_content", None)
            if m.get("name", None) is not None:
                msg["name"] = m.get("name", None)
            if m.get("tool_call_id", None) is not None:
                msg["tool_call_id"] = m.get("tool_call_id", None)
            if m.get("tool_calls", None) is not None:
                msg["tool_calls"] = _normalize_tool_calls(m.get("tool_calls", None))
            chat.append(msg)

    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    if tools:
        kwargs["tools"] = _to_plain(tools)
    return TOKENIZER.apply_chat_template(chat, **kwargs)


# SGLang /generate 接口支持的 sampling_params 白名单
# 不在此列表中的 OpenAI 额外字段（如 output_config）会被过滤掉
SGLANG_SAMPLING_PARAM_KEYS = {
    "max_tokens",
    "max_new_tokens", 
    "min_new_tokens",
    "stop", 
    "stop_token_ids",
    "temperature", 
    "top_p", 
    "top_k",
    "min_p", 
    "frequency_penalty", 
    "presence_penalty", 
    "repetition_penalty",
    "ignore_eos", 
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "regex", 
    "json_schema", 
    "ebnf",
    "no_stop_trim",
    "logit_bias",
}


def _filter_sampling_params(extra_params: Dict[str, Any] | None) -> Dict[str, Any]:
    """从 extra_params 中只保留 SGLang sampling_params 支持的字段。"""
    if not extra_params:
        return {}
    filtered = {k: v for k, v in extra_params.items() if k in SGLANG_SAMPLING_PARAM_KEYS}
    dropped = set(extra_params) - set(filtered)
    if dropped:
        logger.warning(f"Dropped unsupported extra params: {dropped=}, {filtered=}")
    return filtered


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

    # 透传额外参数到 sampling_params（只保留 SGLang 支持的字段）
    payload["sampling_params"].update(_filter_sampling_params(extra_params))

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

    # 透传额外参数到 sampling_params（只保留 SGLang 支持的字段）
    payload["sampling_params"].update(_filter_sampling_params(extra_params))

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
    traj_id = trajectory_store.traj_id
    parquet_fp = os.path.join(PARQUET_PATH, f"{traj_id}.parquet")
    jsonl_fp = os.path.join(JSON_PATH, f"{traj_id}.jsonl")
    trajectory_store.save_parquet(parquet_fp, TOKENIZER)
    trajectory_store.save_jsonl(jsonl_fp)
    logger.info("trajectory saved: traj_id=%s, agents=%d", traj_id, len(trajectory_store.agents))



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

    store = trajectory_store

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
        created_ts = int(time.time())

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
                text = token_ids_to_text(ids).rstrip('\ufffd')
                delta = text[len(prev_text) :]
                prev_text = text
                if delta:
                    yield sse_format(
                        {
                            "id": cmpl_id,
                            "object": "text_completion",
                            "created": created_ts,
                            "model": body.model,
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
                    "created": created_ts,
                    "model": body.model,
                    "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                }
            )

            # Completions are not multi-turn; create a new agent per request
            synthetic_msgs = [{"role": "user", "content": prompt}]
            response_msg = {"role": "assistant", "content": prev_text}
            agent = store.find_or_create_agent(synthetic_msgs, body.tools)
            agent.append_turn(synthetic_msgs, response_msg, input_ids, full_ids)
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

    synthetic_msgs = [{"role": "user", "content": prompt}]
    response_msg = {"role": "assistant", "content": output_text}
    agent = store.find_or_create_agent(synthetic_msgs, body.tools)
    agent.append_turn(synthetic_msgs, response_msg, input_ids, output_ids)
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

    store = trajectory_store
    incoming_msgs = [m.model_dump() for m in body.messages]
    agent = store.get_agent(incoming_msgs, body.tools)

    if agent is not None and agent.messages and agent.prompt_token_ids and agent.response_token_ids:
        prompt = build_chat_prompt(body.messages, tools=body.tools)
        input_ids = text_to_token_ids(prompt)
        prompt_prefix = build_chat_prompt(agent.messages, tools=body.tools, add_generation_prompt=False)
        input_ids_prefix = text_to_token_ids(prompt_prefix)
        tool_ids = input_ids[len(input_ids_prefix): ]
        input_ids = agent.prompt_token_ids + agent.response_token_ids + tool_ids
    else:
        prompt = build_chat_prompt(body.messages, tools=body.tools)
        input_ids = text_to_token_ids(prompt)
        tool_ids = []

    # Determine whether to preserve special tokens for tool parsing
    skip_sp = not (
        TOOL_PARSER
        and body.tools
        and getattr(TOOL_PARSER, "requires_no_skip_special_tokens", False)
    )
    use_tool_parser = TOOL_PARSER is not None and body.tools

    # Create per-request tool parser copy to avoid concurrent state corruption
    if TOOL_PARSER:
        req_parser = copy.copy(TOOL_PARSER)
        req_parser.prev_tool_call_arr = []
        req_parser.streamed_args_for_tool = []
        req_parser._reasoning_parser = None
        req_parser._no_reasoning_prev_len = 0
        if hasattr(req_parser, '_reset_streaming_state'):
            req_parser._reset_streaming_state()
        req_parser.init_reasoning(prompt)
    else:
        req_parser = None

    if body.stream:
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_ts = int(time.time())

        async def generator():
            full_ids = []
            prev_text = ""
            prev_ids: List[int] = []
            has_tool_calls = False
            # For tool parser: track content-only text (without reasoning block)
            tool_prev_text = ""
            tool_prev_ids: List[int] = []
            # Track delta for no-parser fallback
            no_parser_prev_len = 0

            # OpenAI spec: first chunk must include role
            yield sse_format(
                {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": body.model,
                    "choices": [
                        {
                            "delta": {"role": "assistant"},
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                }
            )

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
                text = token_ids_to_text(ids, skip_special_tokens=skip_sp).rstrip('\ufffd')
                delta_ids = ids[len(prev_ids):]

                # Extract reasoning vs content via tool parser
                if req_parser:
                    reasoning_delta, content_delta = req_parser.process_reasoning_delta(text)
                else:
                    reasoning_delta = None
                    content_delta = text[no_parser_prev_len:]
                    no_parser_prev_len = len(text)
                    content_delta = content_delta if content_delta else None

                prev_text = text
                prev_ids = list(ids)

                # Emit reasoning_content delta
                if reasoning_delta:
                    yield sse_format(
                        {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": body.model,
                            "choices": [
                                {
                                    "delta": {"reasoning_content": reasoning_delta},
                                    "index": 0,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )

                # Emit content delta (with or without tool parsing)
                if content_delta:
                    if use_tool_parser and req_parser:
                        content_text = req_parser.get_content_text(text)
                        msg = req_parser.extract_tool_calls_streaming(
                            previous_text=tool_prev_text,
                            current_text=content_text,
                            delta_text=content_delta,
                            previous_token_ids=tool_prev_ids,
                            current_token_ids=ids,
                            delta_token_ids=delta_ids,
                            request=body,
                        )
                        tool_prev_text = content_text
                        tool_prev_ids = list(ids)

                        if msg is not None:
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
                                        "created": created_ts,
                                        "model": body.model,
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
                        yield sse_format(
                            {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": body.model,
                                "choices": [
                                    {
                                        "delta": {"content": content_delta},
                                        "index": 0,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )

            # Flush any remaining tool parser buffer after stream ends
            # (e.g. </tool_call> was the last token and never got processed)
            if use_tool_parser and req_parser:
                content_text = req_parser.get_content_text(prev_text)
                while True:
                    flush_msg = req_parser.extract_tool_calls_streaming(
                        previous_text=tool_prev_text,
                        current_text=content_text,
                        delta_text="",
                        previous_token_ids=tool_prev_ids,
                        current_token_ids=prev_ids,
                        delta_token_ids=[],
                        request=body,
                    )
                    if flush_msg is None:
                        break
                    flush_delta: Dict[str, Any] = {}
                    if flush_msg.content:
                        flush_delta["content"] = flush_msg.content
                    if flush_msg.tool_calls:
                        has_tool_calls = True
                        flush_delta["tool_calls"] = [
                            tc.model_dump(exclude_none=True) for tc in flush_msg.tool_calls
                        ]
                    if flush_delta:
                        yield sse_format(
                            {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": body.model,
                                "choices": [
                                    {
                                        "delta": flush_delta,
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
                    "created": created_ts,
                    "model": body.model,
                    "choices": [
                        {
                            "delta": {},
                            "index": 0,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            )

            # Build response message for trajectory storage
            incoming_msgs = [m.model_dump() for m in body.messages]
            if req_parser and body.tools:
                reasoning_text, content_text = req_parser.extract_reasoning(prev_text)
                extracted = req_parser.extract_tool_calls(content_text, body)
                if extracted.tools_called and extracted.tool_calls:
                    content_text = extracted.content
                    tool_calls = [
                        tc.model_dump() for tc in extracted.tool_calls
                    ]
                else:
                    tool_calls = None

                # NOTE: only save agent when tools are involved, excluding title naming agent
                # content_text = req_parser.post_process_content(content_text)
                response_msg: Dict[str, Any] = {"role": "assistant", "content": content_text}
                if reasoning_text is not None:
                    response_msg["reasoning_content"] = reasoning_text
                if tool_calls:
                    response_msg["tool_calls"] = tool_calls

                incoming_msgs = [m.model_dump() for m in body.messages]
                agent = store.find_or_create_agent(incoming_msgs, body.tools)
                agent.append_turn(incoming_msgs, response_msg, input_ids, full_ids, tool_ids)
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

    # Extract reasoning content (handled by tool parser)
    if req_parser:
        reasoning_content, content_text = req_parser.extract_reasoning(output_text)
    else:
        reasoning_content, content_text = None, output_text

    # Try to extract tool calls from content (reasoning already stripped)
    message: Dict[str, Any] = {"role": "assistant", "content": content_text}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    finish_reason = "stop"

    if req_parser and body.tools:
        extracted = req_parser.extract_tool_calls(content_text, body)
        if extracted.tools_called and extracted.tool_calls:
            finish_reason = "tool_calls"
            message["content"] = extracted.content
            message["tool_calls"] = [
                tc.model_dump() for tc in extracted.tool_calls
            ]

    incoming_msgs = [m.model_dump() for m in body.messages]
    agent = store.find_or_create_agent(incoming_msgs, body.tools)
    agent.append_turn(incoming_msgs, message, input_ids, output_ids, tool_ids)
    save_trajectory()

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
        choices=["qwen3_coder", "deepseek_v32", "glm47", "kimi_k2"],
        help="Tool call parser to use for extracting tool calls from model output",
    )
    return parser.parse_args()


def init_globals(args):
    global SGLANG_URL, SGLANG_BASE_URL, PARQUET_PATH, JSON_PATH, TOKENIZER, TOOL_PARSER, trajectory_store
    SGLANG_BASE_URL = args.sglang_base_url
    SGLANG_URL = f"{SGLANG_BASE_URL}/generate"
    PARQUET_PATH = args.parquet_path
    JSON_PATH = args.json_path

    # 确保数据目录存在
    for path in [PARQUET_PATH, JSON_PATH]:
        os.makedirs(path, exist_ok=True)

    logger.info("Loading tokenizer from %s ...", args.tokenizer_path)
    TOKENIZER = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    trajectory_store = TrajectoryStore()
    logger.info("SGLang URL: %s, base: %s", SGLANG_URL, SGLANG_BASE_URL)
    logger.info("Trajectory store initialized: traj_id=%s", trajectory_store.traj_id)

    # Initialize tool parser
    tool_parser_name = getattr(args, "tool_parser", None)
    if tool_parser_name == "qwen3_coder":
        from tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
        TOOL_PARSER = Qwen3CoderToolParser(TOKENIZER)
        logger.info("Tool parser: Qwen3CoderToolParser")
    elif tool_parser_name == "deepseek_v32":
        from tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
        TOOL_PARSER = DeepSeekV32ToolParser(TOKENIZER)
        logger.info("Tool parser: DeepSeekV32ToolParser")
    elif tool_parser_name == "glm47":
        from tool_parsers.glm47_moe_tool_parser import Glm47MoeToolParser
        TOOL_PARSER = Glm47MoeToolParser(TOKENIZER)
        logger.info("Tool parser: Glm47MoeToolParser")
    elif tool_parser_name == "kimi_k2":
        from tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser
        TOOL_PARSER = KimiK2ToolParser(TOKENIZER)
        logger.info("Tool parser: KimiK2ToolParser")
    else:
        logger.warning(f"Tool parser must be defined: {tool_parser_name=}")
        TOOL_PARSER = None


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    init_globals(args)
    uvicorn.run(app, host=args.host, port=args.port)
