import json
import uuid
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ===== 数据结构 =====
class Message(BaseModel):
    role: str
    # content 可以是字符串、多模态内容列表、或 None（如 tool_calls 消息）
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    reasoning_content: Optional[str] = None
    # tool calling 相关字段
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    # function calling（旧版）/ 消息来源名称
    name: Optional[str] = None

    class Config:
        extra = "allow"  # 允许未知字段透传，避免客户端发送新字段时 422


class OpenAICompletionRequest(BaseModel):
    model: str
    # prompt 可以是字符串、字符串列表、token ID 列表、或嵌套 token ID 列表
    prompt: Union[str, List[str], List[int], List[List[int]]] = ""
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # 新版 API 字段，与 max_tokens 二选一
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    n: int = 1
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    logprobs: Optional[int] = None
    echo: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None

    class Config:
        extra = "allow"


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # 新版 API 字段
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    n: int = 1
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None

    # tool calling
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None

    # NOTE: DEPRECATED OpenAI API, DO NOT USE THESE FIELDS
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    user: Optional[str] = None

    class Config:
        extra = "allow"


# ======= 存储结构 =======


def _normalize_message_for_comparison(msg: Dict[str, Any]) -> tuple:
    """Normalize a message dict to a comparable tuple for prefix matching."""
    role = msg.get("role", "")
    content = msg.get("content")
    if content is None:
        content = ""
    elif isinstance(content, list):
        # Multi-modal content: extract text parts
        parts = []
        if isinstance(content, str):
            parts.append(content)
        else:
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
        content = "\n\n".join(parts)
    tool_call_id = msg.get("tool_call_id", "")
    tool_calls = msg.get("tool_calls")
    tool_calls_key = json.dumps(tool_calls, sort_keys=True) if tool_calls else ""
    # return (role, content, tool_call_id, tool_calls_key)
    return (role, content)


def _count_segments(arr):
    if not arr:
        return 0
    count = 1
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            count += 1
    return count


class AgentTrajectory:
    def __init__(self, agent_id: str = None, input_tools: Optional[List[Dict[str, Any]]] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.messages: List[Dict[str, Any]] = []  # Full conversation history (OpenAI format)
        self.prompt_token_ids: List[int] = []  # Per-turn input token IDs
        self.response_token_ids: List[int] = []  # Per-turn output token IDs
        self.response_mask: List[int] = []

        self.create_time = datetime.now().isoformat()
        self.update_time = ""
        self.finish_time = ""
        self.tools = input_tools
        self.is_fresh = True

    def matches_prefix(self, incoming_messages: List[Dict[str, Any]]) -> bool:
        """Return True if self.messages is a non-empty prefix of incoming_messages."""
        if not self.messages:
            return False
        if len(self.messages) > len(incoming_messages):
            return False
        if incoming_messages and incoming_messages[-1].get("role", "") == "user":
            return False

        for stored, incoming in zip(self.messages, incoming_messages):
            stored_norm = _normalize_message_for_comparison(stored)
            incoming_norm = _normalize_message_for_comparison(incoming)
            if stored_norm != incoming_norm:
                logger.debug(f"Prefix mismatch for agent {self.agent_id}:\n[STORED]\n{stored_norm}\n[IMCOMING]\n{incoming_norm}")
                return False
        return True

    def append_turn(
        self,
        full_messages: List[Dict[str, Any]],
        response_message: Dict[str, Any],
        input_ids: List[int],
        output_ids: List[int],
        tool_ids: List[int] = [],
        tokenizer = None
    ):
        """Append a new turn: update messages to full history + response, record token IDs."""
        if self.prompt_token_ids and tool_ids:
            # Tool tokens do not have token mismatch issue
            self.response_token_ids.extend(tool_ids)
            self.response_mask.extend([1] * len(tool_ids))
        else:
            self.prompt_token_ids.extend(input_ids)

        self.messages = list(full_messages) + [response_message]
        self.response_token_ids.extend(output_ids)
        self.response_mask.extend([0] * len(output_ids))
        self.update_time = datetime.now().isoformat()
        self.is_fresh = True

    def to_jsonl_dict(self) -> Dict[str, Any]:
        """For JSONL export: agent_id + messages only, no token IDs."""
        self.finish_time = datetime.now().isoformat()
        num_turns = _count_segments(self.response_mask)
        retdata = {
            "agent_id": self.agent_id,
            "messages": self.messages,
            "metadata": {
                "num_turns": num_turns,
                "total_prompt_tokens": len(self.prompt_token_ids),
                "total_obs_tokens": sum(self.response_mask),
                "total_resp_tokens": len(self.response_token_ids) - sum(self.response_mask),
                "total_agent_tokens": len(self.response_token_ids),
                "create_time": self.create_time,
                "update_time": self.update_time,
                "finish_time": self.finish_time,
            }
        }
        if self.tools:
            retdata["tools"] = self.tools
        self.is_fresh = False
        return retdata

    def to_parquet_dict(self, tokenizer=None) -> Dict[str, Any]:
        """For parquet export: agent_id + messages + token IDs (as JSON strings)."""
        retdata = {
            "agent_id": self.agent_id,
            "prompt_token_ids": self.prompt_token_ids,
            "response_token_ids": self.response_token_ids,
            "response_mask": self.response_mask,
        }
        if tokenizer:
            prompt_text = tokenizer.decode(self.prompt_token_ids, skip_special_tokens=False)
            response_text = tokenizer.decode(self.response_token_ids, skip_special_tokens=False)
            retdata["prompt_text"] = prompt_text
            retdata["response_text"] = response_text
        self.is_fresh = False
        return retdata


class TrajectoryStore:
    def __init__(self, traj_id: str = None):
        self.traj_id = traj_id or str(uuid.uuid4())
        self.agents: List[AgentTrajectory] = []

    def find_or_create_agent(
        self, 
        incoming_messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AgentTrajectory:
        """Find an agent whose messages are a prefix of incoming_messages, or create a new one."""
        for agent in self.agents:
            if agent.matches_prefix(incoming_messages):
                return agent

        new_agent = AgentTrajectory(agent_id=None, input_tools=tools)
        self.agents.append(new_agent)
        logger.info(
            "traj %s: created new agent %s (total agents: %d)",
            self.traj_id, new_agent.agent_id, len(self.agents),
        )
        return new_agent

    def get_agent(
        self, 
        incoming_messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AgentTrajectory:
        """Find an agent whose messages are a prefix of incoming_messages, or create a new one."""
        for agent in self.agents:
            if agent.matches_prefix(incoming_messages):
                return agent
        return None

    def save_jsonl(self, path: str):
        """Export to JSONL: one line per agent, messages only, ordered."""
        with open(path, "w", encoding="utf-8") as f:
            for agent in self.agents:
                f.write(json.dumps(agent.to_jsonl_dict(), ensure_ascii=False) + "\n")

    def save_parquet(self, path: str, tokenizer=None):
        """Export to parquet: one row per agent, with messages and token IDs, ordered."""
        rows = [agent.to_parquet_dict(tokenizer) for agent in self.agents]
        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(path, index=False)