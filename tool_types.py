"""
Standalone data types and base class for tool call parsing.
Replaces vllm protocol types with minimal Pydantic models.
"""

import uuid
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Optional

from pydantic import BaseModel, Field


def _generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


class FunctionCall(BaseModel):
    name: str
    arguments: str

    class Config:
        extra = "allow"


class ToolCall(BaseModel):
    id: str = Field(default_factory=_generate_tool_call_id)
    type: str = "function"
    function: FunctionCall

    class Config:
        extra = "allow"


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class DeltaToolCall(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    index: int = 0
    function: Optional[DeltaFunctionCall] = None


class ExtractedToolCallInformation(BaseModel):
    tools_called: bool = False
    tool_calls: list[ToolCall] = Field(default_factory=list)
    content: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[DeltaToolCall]] = None


class ToolParser:
    """
    Base class for tool call parsers.
    Subclasses implement extract_tool_calls (non-streaming)
    and extract_tool_calls_streaming (streaming).
    """

    def __init__(
        self,
        tokenizer: Any,
        tools: list[dict] | None = None,
    ):
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.model_tokenizer = tokenizer
        self.tools: list[dict] = tools or []

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()

    def extract_tool_calls(
        self, model_output: str, request: Any
    ) -> ExtractedToolCallInformation:
        raise NotImplementedError

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: Any,
    ) -> DeltaMessage | None:
        raise NotImplementedError
