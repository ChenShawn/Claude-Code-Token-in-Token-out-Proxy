"""
Reasoning content extractor for <think>...</think> tags.

Extracts reasoning content from model output into a separate field,
supporting both streaming (stateful) and non-streaming (one-shot) modes.
"""

import re
from enum import Enum
from typing import Optional, Tuple


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_PATTERN = re.compile(r"^<think>(.*?)</think>(.*)", re.DOTALL)
_THINK_PATTERN_NO_OPEN = re.compile(r"^(.*?)</think>(.*)", re.DOTALL)


class ReasoningParser:
    """Extracts <think>...</think> reasoning from model output.

    Non-streaming: use the static method `extract_reasoning()`.
    Streaming: create an instance and call `process_delta()` for each chunk.
    """

    class _State(Enum):
        DETECT = "detect"        # Haven't decided if output starts with <think>
        IN_THINK = "in_think"    # Inside <think> block, accumulating reasoning
        CONTENT = "content"      # Past <think> block, emitting normal content

    @staticmethod
    def extract_reasoning(text: str, starts_in_think: bool = False) -> Tuple[Optional[str], str]:
        """One-shot extraction for non-streaming responses.

        Args:
            text: The full model output text.
            starts_in_think: If True, the prompt already opened a <think> block,
                so the output starts directly with reasoning (no <think> prefix).

        Returns:
            (reasoning_content, content) — reasoning_content is None if no think block found.
        """
        if starts_in_think:
            # Strip redundant <think> tag — the prompt already opened one
            work_text = text
            if work_text.startswith(_THINK_OPEN):
                work_text = work_text[len(_THINK_OPEN):]
                if work_text.startswith("\n"):
                    work_text = work_text[1:]
            m = _THINK_PATTERN_NO_OPEN.match(work_text)
            if m:
                reasoning = m.group(1).strip("\n")
                content = m.group(2).lstrip("\n")
                return reasoning, content
            # No </think> found — model likely skipped reasoning entirely
            return None, text

        m = _THINK_PATTERN.match(text)
        if m:
            reasoning = m.group(1).strip("\n")
            content = m.group(2).lstrip("\n")
            return reasoning, content
        return None, text

    def __init__(self, starts_in_think: bool = False):
        self._starts_in_think = starts_in_think
        self._think_open_len = 0 if starts_in_think else len(_THINK_OPEN)
        if starts_in_think:
            self._state = self._State.IN_THINK
        else:
            self._state = self._State.DETECT
        self._prev_len = 0          # Length of text already processed
        self._think_close_pos = -1  # Position of '<' in </think> tag
        self._think_end_pos = -1    # Position after </think> (and optional \n), where content starts
        self._reasoning_emitted = 0 # How much reasoning text we've already emitted
        self._pending_newline_skip = False  # Need to skip \n after </think> on next content call

    def process_delta(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Process the cumulative decoded text and return deltas.

        Args:
            full_text: The entire decoded output so far (cumulative).

        Returns:
            (reasoning_delta, content_delta) — at most one is non-None per call.
            Both can be None if we're still buffering/detecting.
        """
        if self._state == self._State.DETECT:
            return self._handle_detect(full_text)
        elif self._state == self._State.IN_THINK:
            return self._handle_in_think(full_text)
        else:  # CONTENT
            return self._handle_content(full_text)

    def _handle_detect(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Buffer text until we can determine if it starts with <think>."""
        # Not enough text yet to decide
        if len(full_text) < len(_THINK_OPEN):
            # Check if what we have is still a valid prefix of <think>
            if _THINK_OPEN.startswith(full_text):
                return None, None
            else:
                # Definitely not a <think> block
                self._state = self._State.CONTENT
                self._prev_len = 0
                return self._handle_content(full_text)

        # We have enough text to check
        if full_text.startswith(_THINK_OPEN):
            self._state = self._State.IN_THINK
            self._reasoning_emitted = 0
            return self._handle_in_think(full_text)
        else:
            self._state = self._State.CONTENT
            self._prev_len = 0
            return self._handle_content(full_text)

    def _handle_in_think(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Inside <think> block — emit reasoning deltas."""
        # When starts_in_think, detect and skip redundant <think> tag
        if (self._starts_in_think
                and self._think_open_len == 0
                and self._reasoning_emitted == 0):
            if len(full_text) < len(_THINK_OPEN):
                # Not enough text — check if it could still be a <think> prefix
                if _THINK_OPEN.startswith(full_text):
                    return None, None
            elif full_text.startswith(_THINK_OPEN):
                new_offset = len(_THINK_OPEN)
                if new_offset < len(full_text) and full_text[new_offset] == "\n":
                    new_offset += 1
                self._think_open_len = new_offset

        close_pos = full_text.find(_THINK_CLOSE)
        if close_pos == -1:
            # Still inside <think>, emit reasoning delta
            # Hold back up to len("</think>")-1 chars to avoid emitting partial close tag
            reasoning_text = full_text[self._think_open_len:]
            safe_len = max(0, len(reasoning_text) - (len(_THINK_CLOSE) - 1))
            delta = reasoning_text[self._reasoning_emitted:safe_len]
            self._reasoning_emitted = safe_len
            self._prev_len = len(full_text)
            return (delta if delta else None, None)
        else:
            # Found </think> — emit final reasoning delta and transition
            reasoning_text = full_text[self._think_open_len:close_pos]
            reasoning_delta = reasoning_text[self._reasoning_emitted:]
            self._reasoning_emitted = len(reasoning_text)

            self._think_close_pos = close_pos
            self._think_end_pos = close_pos + len(_THINK_CLOSE)
            # Skip newline after </think> if present
            if self._think_end_pos < len(full_text) and full_text[self._think_end_pos] == "\n":
                self._think_end_pos += 1
            else:
                # Newline might arrive in the next chunk
                self._pending_newline_skip = True
            self._state = self._State.CONTENT
            self._prev_len = self._think_end_pos

            # There might be content after </think> in this same chunk
            content_text = full_text[self._think_end_pos:]
            if content_text:
                self._prev_len = len(full_text)
                return (reasoning_delta if reasoning_delta else None, content_text)

            return (reasoning_delta if reasoning_delta else None, None)

    def _handle_content(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Past <think> block — emit content deltas."""
        if self._pending_newline_skip and self._prev_len < len(full_text):
            if full_text[self._prev_len] == "\n":
                self._prev_len += 1
                self._think_end_pos += 1
            self._pending_newline_skip = False
        delta = full_text[self._prev_len:]
        self._prev_len = len(full_text)
        return (None, delta if delta else None)

    @property
    def has_reasoning(self) -> bool:
        """Whether a <think> block was detected."""
        return self._state != self._State.DETECT and self._reasoning_emitted > 0

    def get_content_text(self, full_text: str) -> str:
        """Get the content-only portion of the full text (for tool parser use)."""
        if self._think_end_pos > 0:
            return full_text[self._think_end_pos:].lstrip("\n")
        if self._state == self._State.IN_THINK:
            return ""  # Still in think block, no content yet
        return full_text

    def get_reasoning_text(self, full_text: str) -> Optional[str]:
        """Get the full reasoning text from the full output."""
        if self._think_close_pos >= 0:
            return full_text[self._think_open_len:self._think_close_pos].strip("\n")
        if self._state == self._State.IN_THINK:
            return full_text[self._think_open_len:].strip("\n")
        return None
