"""
Anthropic-compatible LLM Gateway Proxy.

Supports two upstream backends (controlled by UPSTREAM_TYPE env var):
  - "aws" (default): AWS Bedrock invoke, base64-wrapped stream chunks
  - "google":  Google Vertex AI rawPredict, standard Anthropic SSE stream

Environment variables:
  UPSTREAM_TYPE        - "aws" or "google" (default: "aws")
  BEDROCK_UPSTREAM_BASE - Bedrock base URL
  GOOGLE_UPSTREAM_BASE  - Google Vertex base URL
  UPSTREAM_API_KEY      - Optional default API key for upstream
  SAVE_DIR              - Directory to save request/response logs (default: /data/logs)
  SAVE_ALL_REQUESTS     - "true" to save successful (200) requests too (default: only errors)

Usage:
    UPSTREAM_TYPE=aws \
    SAVE_DIR=/data/logs \
        uvicorn proxy:app --host 0.0.0.0 --port 8080
"""

import asyncio
import json
import logging
import os
import time
import uuid
import base64
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, List, Optional

import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response

app = FastAPI(title="Anthropic-compatible LLM Gateway", redirect_slashes=False)

# ── Configuration (all from env) ─────────────────────────────────────────

UPSTREAM_TYPE = os.environ.get("UPSTREAM_TYPE", "google").lower()  # "aws" or "google"

# Bedrock config
BEDROCK_BASE = os.environ.get(
    "BEDROCK_UPSTREAM_BASE",
    "https://runway.devops.rednote.life/openai/bedrock_runtime/model",
)
BEDROCK_INVOKE_URL = f"{BEDROCK_BASE.rstrip('/')}/invoke"
BEDROCK_STREAM_URL = f"{BEDROCK_BASE.rstrip('/')}/invoke-with-response-stream"
BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Google Vertex config
GOOGLE_BASE = os.environ.get(
    "GOOGLE_UPSTREAM_BASE",
    "https://runway.devops.rednote.life/openai/google/anthropic/v1",
)
GOOGLE_RAW_URL = f"{GOOGLE_BASE.rstrip('/')}:rawPredict"
GOOGLE_STREAM_URL = f"{GOOGLE_BASE.rstrip('/')}:streamRawPredict"
GOOGLE_ANTHROPIC_VERSION = "vertex-2023-10-16"

# Auth key (used as api-key for Google, token for Bedrock)
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", "")

SAVE_DIR = os.environ.get("SAVE_DIR", "/data/logs")
SAVE_ALL_REQUESTS = os.environ.get("SAVE_ALL_REQUESTS", "1").lower() in ("1", "true", "yes")

_HOP_BY_HOP_HEADERS = frozenset({
    "transfer-encoding", "content-encoding", "connection",
    "keep-alive", "upgrade", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "content-length",
})

_http_client: Optional[httpx.AsyncClient] = None

logger = logging.getLogger(__name__)

print(f"[proxy] Upstream: {UPSTREAM_TYPE}, save_all={SAVE_ALL_REQUESTS}, save_dir={SAVE_DIR}")


# ── TrajectoryStore (adapted from cc-tito-proxy/global_types.py) ─────────

def _normalize_message_for_comparison(msg: Dict[str, Any]) -> tuple:
    """Normalize a message dict to a comparable tuple for prefix matching."""
    role = msg.get("role", "")
    content = msg.get("content")
    if content is None:
        content = ""
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        content = "\n\n".join(parts)
    return (role, content)


class AgentTrajectory:
    def __init__(self, agent_id: str = None, input_tools: Optional[List[Dict[str, Any]]] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.messages: List[Dict[str, Any]] = []
        self.create_time = datetime.now().isoformat()
        self.update_time = datetime.now().isoformat()
        self.tools = input_tools
        self._num_turns = 0

    def matches_prefix(self, incoming_messages: List[Dict[str, Any]]) -> bool:
        """Return True if self.messages is a non-empty prefix of incoming_messages."""
        if not self.messages:
            return False
        if len(self.messages) > len(incoming_messages):
            return False
        if incoming_messages and incoming_messages[-1].get("role", "") == "user":
            return False
        for stored, incoming in zip(self.messages, incoming_messages):
            if _normalize_message_for_comparison(stored) != _normalize_message_for_comparison(incoming):
                return False
        return True

    def append_turn(self, full_messages: List[Dict[str, Any]], response_message: Dict[str, Any]):
        """Append a new turn: set messages to full history + response."""
        self.messages = list(full_messages) + [response_message]
        self._num_turns += 1
        self.update_time = datetime.now().isoformat()

    def to_jsonl_dict(self) -> Dict[str, Any]:
        retdata = {
            "agent_id": self.agent_id,
            "messages": self.messages,
            "metadata": {
                "num_turns": self._num_turns,
                "create_time": self.create_time,
                "update_time": self.update_time,
            },
        }
        if self.tools:
            retdata["tools"] = self.tools
        return retdata


class TrajectoryStore:
    def __init__(self, traj_id: str = None):
        self.traj_id = traj_id or str(uuid.uuid4())
        self.agents: List[AgentTrajectory] = []

    def find_or_create_agent(
        self,
        incoming_messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentTrajectory:
        for agent in self.agents:
            if agent.matches_prefix(incoming_messages):
                return agent
        new_agent = AgentTrajectory(agent_id=None, input_tools=tools)
        self.agents.append(new_agent)
        logger.info("traj %s: new agent %s (total: %d)", self.traj_id, new_agent.agent_id, len(self.agents))
        return new_agent

    def save_jsonl(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for agent in self.agents:
                f.write(json.dumps(agent.to_jsonl_dict(), ensure_ascii=False) + "\n")


# Per-token trajectory stores (in-memory)
_trajectory_stores: Dict[str, TrajectoryStore] = {}


def _reconstruct_assistant_message(output, stream: bool) -> Optional[Dict[str, Any]]:
    """Reconstruct an Anthropic-format assistant message from proxy output."""
    if output is None:
        return None
    if not stream:
        # Non-stream: output is full API response dict with "content" field
        if isinstance(output, dict) and "content" in output:
            return {"role": "assistant", "content": output["content"]}
        return None
    # Stream: output is merged {thinking, text, tool_uses}
    if not isinstance(output, dict):
        return None
    content = []
    thinking = output.get("thinking", "")
    if thinking:
        content.append({"type": "thinking", "thinking": thinking})
    text = output.get("text", "")
    if text:
        content.append({"type": "text", "text": text})
    for tu in output.get("tool_uses", []):
        content.append({"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": tu["input"]})
    if not content:
        return None
    return {"role": "assistant", "content": content}


@app.on_event("startup")
async def _startup():
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=300.0))


@app.on_event("shutdown")
async def _shutdown():
    global _http_client
    if _http_client:
        await _http_client.aclose()


# ── Payload preparation ──────────────────────────────────────────────────

def _strip_cache_control_scope(obj):
    """Remove 'scope' from all cache_control objects (upstream often doesn't support scope: global)."""
    if isinstance(obj, dict):
        cc = obj.get("cache_control")
        if isinstance(cc, dict):
            cc.pop("scope", None)
        for v in obj.values():
            _strip_cache_control_scope(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_cache_control_scope(item)


def _has_defer_loading(tools) -> bool:
    if not isinstance(tools, list):
        return False
    return any(isinstance(t, dict) and t.get("defer_loading") for t in tools)


def _convert_tool_reference_to_text(obj):
    """Convert tool_reference blocks to text (Bedrock doesn't support tool_reference)."""
    if isinstance(obj, dict):
        if obj.get("type") == "tool_reference":
            obj["type"] = "text"
            obj["text"] = f"[Tool reference: {obj.get('tool_name', 'unknown')}]"
            obj.pop("tool_name", None)
        for v in obj.values():
            _convert_tool_reference_to_text(v)
    elif isinstance(obj, list):
        for item in obj:
            _convert_tool_reference_to_text(item)


def _prepare_bedrock_payload(body: dict) -> dict:
    """Bedrock: strip model/stream, add anthropic_version + beta flags, fix unsupported fields."""
    payload = json.loads(json.dumps(body))
    payload.pop("model", None)
    payload.pop("stream", None)
    payload["anthropic_version"] = BEDROCK_ANTHROPIC_VERSION

    beta = payload.get("anthropic_beta", [])
    if not isinstance(beta, list):
        beta = []
    if "fine-grained-tool-streaming-2025-05-14" not in beta:
        beta.append("fine-grained-tool-streaming-2025-05-14")
    if "context_management" in payload and "context-management-2025-06-27" not in beta:
        beta.append("context-management-2025-06-27")
    if (isinstance(payload.get("output_config"), dict)
            and "effort" in payload["output_config"]
            and "effort-2025-11-24" not in beta):
        beta.append("effort-2025-11-24")
    if _has_defer_loading(payload.get("tools", [])) and "tool-search-tool-2025-10-19" not in beta:
        beta.append("tool-search-tool-2025-10-19")
    payload["anthropic_beta"] = beta

    _strip_cache_control_scope(payload)
    _convert_tool_reference_to_text(payload)
    return payload


# ── Stream parsing ───────────────────────────────────────────────────────

def _merge_stream_output(events: list) -> dict:
    """Merge stream events into a single output dict with thinking, text, and tool_uses."""
    thinking_parts = []
    text_parts = []
    tool_use_blocks: dict[int, dict] = {}

    for ev in events:
        ev_type = ev.get("type")
        if ev_type == "content_block_start":
            idx = ev.get("index", 0)
            block = ev.get("content_block") or {}
            if block.get("type") == "tool_use":
                tool_use_blocks[idx] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input_parts": [],
                }
        elif ev_type == "content_block_delta":
            idx = ev.get("index", 0)
            delta = ev.get("delta") or {}
            if delta.get("type") == "thinking_delta":
                thinking_parts.append(delta.get("thinking", ""))
            elif delta.get("type") == "text_delta":
                text_parts.append(delta.get("text", ""))
            elif delta.get("type") == "input_json_delta":
                if idx in tool_use_blocks:
                    tool_use_blocks[idx]["input_parts"].append(delta.get("partial_json", ""))

    tool_uses = []
    for block in tool_use_blocks.values():
        raw_input = "".join(block["input_parts"])
        try:
            parsed_input = json.loads(raw_input) if raw_input else {}
        except Exception:
            parsed_input = raw_input
        tool_uses.append({"id": block["id"], "name": block["name"], "input": parsed_input})

    return {
        "thinking": "".join(thinking_parts),
        "text": "".join(text_parts),
        "tool_uses": tool_uses,
    }


def _decode_bedrock_chunk(raw_line: str) -> Optional[dict]:
    """Decode a Bedrock base64-wrapped stream line."""
    line = raw_line.strip()
    if not line:
        return None
    try:
        envelope = json.loads(line)
    except json.JSONDecodeError:
        return None
    chunk = envelope.get("chunk")
    b64 = chunk.get("bytes") if isinstance(chunk, dict) else None
    if not b64:
        return None
    try:
        return json.loads(base64.b64decode(b64))
    except Exception:
        return None


def _decode_google_sse_line(line: str) -> Optional[dict]:
    """Decode a standard Anthropic SSE line (data: {...})."""
    line = line.strip()
    if not line or not line.startswith("data: "):
        return None
    data_str = line[6:].strip()
    if not data_str:
        return None
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None


# ── Request saving ───────────────────────────────────────────────────────

def sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _save_request_async(body: dict, stream: bool, status_code: int,
                              output=None, token: str = ""):
    """Save request+response in two formats:
    1. raw/<token>/request_<ts>.json  — per-request raw dump
    2. trajectory/<token>/trajectory.jsonl — grouped by agent via prefix matching
    """
    token_dir = token or "unknown"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"

    """
    # ── 1. Raw format (unchanged) ──
    raw_path = os.path.join(SAVE_DIR, "raw", token_dir, f"request_{ts}.json")
    raw_payload = {
        "request": body,
        "output": output if output is not None else "",
        "stream": stream,
        "status_code": status_code,
    }
    raw_data = json.dumps(raw_payload, ensure_ascii=False, indent=2)

    def _write_raw():
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, "w") as f:
            f.write(raw_data)

    await asyncio.to_thread(_write_raw)

    if status_code != 200:
        out_preview = str(output)[:500] if output is not None else ""
        print(f"[non-200] stream={stream} status={status_code}, body={out_preview}")
    """

    # ── 2. Trajectory format (only for successful requests with messages) ──
    if status_code == 200 and isinstance(body, dict) and "messages" in body:
        assistant_msg = _reconstruct_assistant_message(output, stream)
        if assistant_msg:
            messages = body["messages"]
            tools = body.get("tools")

            store = _trajectory_stores.get(token_dir)
            if store is None:
                store = TrajectoryStore()
                _trajectory_stores[token_dir] = store

            agent = store.find_or_create_agent(messages, tools)
            agent.append_turn(messages, assistant_msg)

            traj_id = store.traj_id
            traj_path = os.path.join(SAVE_DIR, f"{traj_id}.jsonl")

            def _write_traj():
                store.save_jsonl(traj_path)

            await asyncio.to_thread(_write_traj)
            print(f"[trajectory] token={token_dir} agents={len(store.agents)} -> {traj_path}")


# ── Routes ───────────────────────────────────────────────────────────────

@app.api_route("/v1/messages/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def messages_handler_with_path(request: Request, path: str,
                                     authorization: Optional[str] = Header(None)):
    return await _messages_handler_impl(request, path, authorization)


@app.api_route("/v1/messages", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def messages_handler(request: Request, authorization: Optional[str] = Header(None)):
    return await _messages_handler_impl(request, "", authorization)


def _extract_token(request: Request, authorization: Optional[str]) -> str:
    """Extract auth token from x-api-key or Authorization: Bearer header."""
    x_api_key = request.headers.get("x-api-key", "").strip()
    bearer = ""
    if authorization and authorization.startswith("Bearer "):
        bearer = authorization.split(" ", 1)[1]
    if x_api_key and x_api_key not in ("not-available", "not-set", "none", "null", ""):
        return x_api_key
    if bearer:
        return bearer
    return x_api_key


async def _messages_handler_impl(request: Request, path: str,
                                 authorization: Optional[str] = Header(None)):
    token = _extract_token(request, authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token (x-api-key or Authorization Bearer)")

    if UPSTREAM_TYPE == "google":
        return await _handle_google(request, path, token)
    else:
        return await _handle_bedrock(request, path, token)


# ── Google Vertex ────────────────────────────────────────────────────────

async def _handle_google(request: Request, path: str, token: str):
    """Google Vertex AI - only supports POST /v1/messages."""
    key = token or UPSTREAM_API_KEY
    headers = {"Content-Type": "application/json", "api-key": key}

    if path:
        raise HTTPException(status_code=501, detail=f"Google Vertex does not support /v1/messages/{path}")

    if request.method != "POST":
        raise HTTPException(status_code=405, detail="Only POST is supported for Google Vertex")

    body = await request.json()
    stream = body.get("stream", False)

    # Google Vertex doesn't support some fields that Claude Code / Bedrock use
    _GOOGLE_STRIP_FIELDS = {"model", "anthropic_beta", "context_management"}
    payload = {k: v for k, v in body.items() if k not in _GOOGLE_STRIP_FIELDS}
    payload["anthropic_version"] = GOOGLE_ANTHROPIC_VERSION
    if stream:
        payload["stream"] = True
    else:
        payload.pop("stream", None)

    upstream_url = GOOGLE_STREAM_URL if stream else GOOGLE_RAW_URL

    if stream:
        return await _handle_google_stream(body, payload, upstream_url, headers, token)
    else:
        resp = await _http_client.post(upstream_url, json=payload, headers=headers, timeout=3600)
        if resp.status_code != 200:
            asyncio.create_task(_save_request_async(body, False, resp.status_code, resp.text, token=token))
            raise HTTPException(status_code=502, detail=f"Upstream error: {resp.status_code} {resp.text}")
        resp_data = resp.json()
        if isinstance(resp_data, dict) and "Error" in resp_data and "type" not in resp_data:
            asyncio.create_task(_save_request_async(body, False, 502, output=resp_data, token=token))
            raise HTTPException(status_code=502, detail=f"Upstream error in 200 body: {resp_data}")
        if SAVE_ALL_REQUESTS:
            asyncio.create_task(_save_request_async(body, False, 200, output=resp_data, token=token))
        return JSONResponse(content=resp_data)


# ── AWS Bedrock ──────────────────────────────────────────────────────────

async def _handle_bedrock(request: Request, path: str, token: str):
    """AWS Bedrock - supports POST /v1/messages and sub-paths (count_tokens etc.)."""
    key = token or UPSTREAM_API_KEY
    headers = {"Content-Type": "application/json", **({"token": key} if key else {})}

    # Sub-paths or non-POST: passthrough
    if path or request.method != "POST":
        upstream_path = f"/v1/messages/{path}" if path else "/v1/messages"
        query = request.url.query
        upstream_url = f"{BEDROCK_BASE.rstrip('/')}{upstream_path}" + (f"?{query}" if query else "")
        fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host",)}
        body_bytes = None
        if request.method in ("POST", "PUT", "PATCH"):
            body_bytes = await request.body()
        resp = await _http_client.request(request.method, upstream_url, content=body_bytes, headers=fwd_headers)
        resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}
        return Response(content=resp.content, status_code=resp.status_code,
                        media_type=resp.headers.get("content-type") or "application/json",
                        headers=resp_headers)

    # POST /v1/messages -> Bedrock invoke
    body = await request.json()
    stream = body.get("stream", False)
    upstream_payload = _prepare_bedrock_payload(body)
    upstream_url = BEDROCK_STREAM_URL if stream else BEDROCK_INVOKE_URL

    if stream:
        return await _handle_bedrock_stream(body, upstream_payload, upstream_url, headers, token)
    else:
        resp = await _http_client.post(upstream_url, json=upstream_payload, headers=headers, timeout=3600)
        if resp.status_code != 200:
            asyncio.create_task(_save_request_async(body, False, resp.status_code, resp.text, token=token))
            raise HTTPException(status_code=502, detail=f"Upstream error: {resp.status_code} {resp.text}")
        resp_data = resp.json()
        if isinstance(resp_data, dict) and "Error" in resp_data and "type" not in resp_data:
            asyncio.create_task(_save_request_async(body, False, 502, output=resp_data, token=token))
            raise HTTPException(status_code=502, detail=f"Upstream error in 200 body: {resp_data}")
        if SAVE_ALL_REQUESTS:
            asyncio.create_task(_save_request_async(body, False, 200, output=resp_data, token=token))
        return JSONResponse(content=resp_data)


# ── Stream handlers ──────────────────────────────────────────────────────

async def _handle_google_stream(body, upstream_payload, upstream_url, headers, token):
    """Google Vertex returns standard Anthropic SSE - passthrough with optional capture."""

    async def raw_generator() -> AsyncGenerator[bytes, None]:
        output_events = [] if SAVE_ALL_REQUESTS else None
        try:
            async with _http_client.stream(
                "POST", upstream_url, json=upstream_payload, headers=headers, timeout=None
            ) as resp:
                if resp.status_code != 200:
                    body_bytes = await resp.aread()
                    try:
                        resp_text = body_bytes.decode("utf-8")
                    except Exception:
                        resp_text = repr(body_bytes)
                    asyncio.create_task(
                        _save_request_async(body, True, resp.status_code, resp_text, token=token)
                    )
                    yield body_bytes
                    return

                async for line in resp.aiter_lines():
                    yield (line + "\n").encode("utf-8")
                    if output_events is not None:
                        event = _decode_google_sse_line(line)
                        if event is not None:
                            output_events.append(event)

                if output_events is not None:
                    merged = _merge_stream_output(output_events)
                    asyncio.create_task(
                        _save_request_async(body, True, 200, output=merged, token=token)
                    )
        except Exception as e:
            yield sse_format("error", {"error": str(e)}).encode()

    return StreamingResponse(
        raw_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _handle_bedrock_stream(body, upstream_payload, upstream_url, headers, token):
    """Bedrock returns base64-wrapped chunks - decode and forward raw bytes."""

    async def raw_generator() -> AsyncGenerator[bytes, None]:
        output_events = [] if SAVE_ALL_REQUESTS else None
        buf = ""
        try:
            async with _http_client.stream(
                "POST", upstream_url, json=upstream_payload, headers=headers, timeout=None
            ) as resp:
                if resp.status_code != 200:
                    body_bytes = await resp.aread()
                    try:
                        resp_text = body_bytes.decode("utf-8")
                    except Exception:
                        resp_text = repr(body_bytes)
                    asyncio.create_task(
                        _save_request_async(body, True, resp.status_code, resp_text, token=token)
                    )
                    yield body_bytes
                    return

                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
                        if output_events is not None:
                            try:
                                buf += chunk.decode("utf-8", errors="replace")
                                while "\n" in buf:
                                    line, buf = buf.split("\n", 1)
                                    event = _decode_bedrock_chunk(line)
                                    if event is not None:
                                        output_events.append(event)
                            except Exception:
                                pass

                if output_events is not None:
                    if buf.strip():
                        event = _decode_bedrock_chunk(buf)
                        if event is not None:
                            output_events.append(event)
                    merged = _merge_stream_output(output_events)
                    asyncio.create_task(
                        _save_request_async(body, True, 200, output=merged, token=token)
                    )
        except Exception as e:
            yield sse_format("error", {"error": str(e)}).encode()

    return StreamingResponse(raw_generator(), media_type="application/octet-stream")
