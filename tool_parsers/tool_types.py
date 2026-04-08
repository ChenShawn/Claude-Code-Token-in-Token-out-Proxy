"""
Standalone data types and base class for tool call parsing.
Replaces vllm protocol types with minimal Pydantic models.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional, Tuple

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .reasoning_parser import ReasoningParser


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

    Optionally supports reasoning extraction (e.g. <think>...</think>).
    Subclasses that need reasoning should set supports_reasoning = True
    and override init_reasoning().
    """

    supports_reasoning: bool = False

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

        self._reasoning_parser: ReasoningParser | None = None
        self._no_reasoning_prev_len: int = 0

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()

    # ---- Reasoning interface ----

    def init_reasoning(self, prompt: str) -> None:
        """Initialize reasoning parser based on the prompt.
        Subclasses override this to create a ReasoningParser.
        Must be called before each request's streaming/non-streaming processing.
        """
        self._reasoning_parser = None
        self._no_reasoning_prev_len = 0

    def extract_reasoning(self, text: str) -> Tuple[Optional[str], str]:
        """Non-streaming: extract reasoning and content from full output text.
        Returns (reasoning_content, content). reasoning_content is None if no reasoning."""
        if self._reasoning_parser:
            return self._reasoning_parser.extract_reasoning(
                text, starts_in_think=self._reasoning_parser._starts_in_think
            )
        return None, text

    def process_reasoning_delta(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Streaming: process cumulative text and return (reasoning_delta, content_delta).
        When no reasoning parser, returns (None, content_delta) with delta computed internally."""
        if self._reasoning_parser:
            return self._reasoning_parser.process_delta(full_text)
        # No reasoning — compute content delta directly
        delta = full_text[self._no_reasoning_prev_len:]
        self._no_reasoning_prev_len = len(full_text)
        return None, delta if delta else None

    def get_content_text(self, full_text: str) -> str:
        """Get the content-only portion (without reasoning block) from full text."""
        if self._reasoning_parser:
            return self._reasoning_parser.get_content_text(full_text)
        return full_text

    def get_reasoning_text(self, full_text: str) -> Optional[str]:
        """Get the full reasoning text, or None if no reasoning block."""
        if self._reasoning_parser:
            return self._reasoning_parser.get_reasoning_text(full_text)
        return None

    # ---- Tool call interface ----

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

    # ---- Model-specific Configuration

    def post_process_content(self, content: str):
        return content
