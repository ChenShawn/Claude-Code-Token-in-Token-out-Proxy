import uuid
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator, Optional, Union


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
class AgentTrajectory:
    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.prompt_token_ids = []
        self.response_token_ids = []
        self.messages = []

    def to_json(self):
        return {
            "agent_id": self.agent_id,
            "messages": self.messages,
            "extra": {
                "num_input_tokens": len(self.prompt_token_ids),
                "num_output_tokens": len(self.response_token_ids),
            }
        }

    def to_parquet_json(self):
        return {
            "agent_id": self.agent_id,
            "num_input_tokens": len(self.prompt_token_ids),
            "num_output_tokens": len(self.response_token_ids),
        }


class TrajectoryStore:
    def __init__(self, traj_id: str = None):
        self.traj_id = traj_id or str(uuid.uuid4())
        self.trajectory_store = []

    def add_trajectory(self, trajectory: AgentTrajectory):
        pass