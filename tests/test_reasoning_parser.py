"""Tests for reasoning_parser.py — <think>...</think> extraction."""

import pytest
from tool_parsers.reasoning_parser import ReasoningParser


# ===== Non-streaming tests =====

class TestExtractReasoning:
    def test_with_think_block(self):
        text = "<think>Let me reason about this.</think>Hello, world!"
        reasoning, content = ReasoningParser.extract_reasoning(text)
        assert reasoning == "Let me reason about this."
        assert content == "Hello, world!"

    def test_with_think_block_newline_after(self):
        text = "<think>Reasoning here.</think>\nActual content."
        reasoning, content = ReasoningParser.extract_reasoning(text)
        assert reasoning == "Reasoning here."
        assert content == "Actual content."

    def test_without_think_block(self):
        text = "Just regular content, no thinking."
        reasoning, content = ReasoningParser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_empty_think_block(self):
        text = "<think></think>Content after empty think."
        reasoning, content = ReasoningParser.extract_reasoning(text)
        assert reasoning == ""
        assert content == "Content after empty think."

    def test_multiline_reasoning(self):
        text = "<think>Line 1\nLine 2\nLine 3</think>Result."
        reasoning, content = ReasoningParser.extract_reasoning(text)
        assert reasoning == "Line 1\nLine 2\nLine 3"
        assert content == "Result."

    def test_think_not_at_start(self):
        text = "Prefix <think>reasoning</think> content"
        reasoning, content = ReasoningParser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_no_content_after_think(self):
        text = "<think>Only reasoning, no content.</think>"
        reasoning, content = ReasoningParser.extract_reasoning(text)
        assert reasoning == "Only reasoning, no content."
        assert content == ""


# ===== Streaming tests =====

class TestStreamingReasoningParser:
    def _feed_incremental(self, full_text: str, chunk_size: int = 3):
        """Feed text to parser in chunks, collecting all deltas."""
        parser = ReasoningParser()
        reasoning_parts = []
        content_parts = []
        for i in range(0, len(full_text), chunk_size):
            chunk_text = full_text[:i + chunk_size]
            r_delta, c_delta = parser.process_delta(chunk_text)
            if r_delta:
                reasoning_parts.append(r_delta)
            if c_delta:
                content_parts.append(c_delta)
        return "".join(reasoning_parts), "".join(content_parts), parser

    def test_streaming_with_think(self):
        text = "<think>I need to think about this carefully.</think>The answer is 42."
        reasoning, content, parser = self._feed_incremental(text, chunk_size=5)
        assert reasoning == "I need to think about this carefully."
        assert content == "The answer is 42."
        assert parser.has_reasoning

    def test_streaming_without_think(self):
        text = "No thinking here, just content."
        reasoning, content, parser = self._feed_incremental(text, chunk_size=5)
        assert reasoning == ""
        assert content == text
        assert not parser.has_reasoning

    def test_streaming_char_by_char(self):
        text = "<think>Reason.</think>Content."
        reasoning, content, parser = self._feed_incremental(text, chunk_size=1)
        assert reasoning == "Reason."
        assert content == "Content."

    def test_streaming_large_chunks(self):
        text = "<think>Reasoning block.</think>Content block."
        reasoning, content, parser = self._feed_incremental(text, chunk_size=100)
        assert reasoning == "Reasoning block."
        assert content == "Content block."

    def test_streaming_empty_think(self):
        text = "<think></think>Only content."
        reasoning, content, _ = self._feed_incremental(text, chunk_size=4)
        # Empty reasoning might produce empty string
        assert content == "Only content."

    def test_streaming_think_newline(self):
        text = "<think>Think.</think>\nContent."
        reasoning, content, _ = self._feed_incremental(text, chunk_size=5)
        assert reasoning == "Think."
        assert content == "Content."

    def test_streaming_no_content_after_think(self):
        text = "<think>Only reasoning.</think>"
        reasoning, content, parser = self._feed_incremental(text, chunk_size=5)
        assert reasoning == "Only reasoning."
        assert content == ""

    def test_get_content_text(self):
        text = "<think>Reasoning.</think>Content here."
        parser = ReasoningParser()
        # Feed all at once
        parser.process_delta(text)
        assert parser.get_content_text(text) == "Content here."
        assert parser.get_reasoning_text(text) == "Reasoning."

    def test_get_content_text_no_think(self):
        text = "Just content."
        parser = ReasoningParser()
        parser.process_delta(text)
        assert parser.get_content_text(text) == text
        assert parser.get_reasoning_text(text) is None

    def test_detect_state_partial_prefix(self):
        """Test that partial '<thi' doesn't cause issues."""
        parser = ReasoningParser()
        # Feed "<thi" — still could be <think>
        r, c = parser.process_delta("<thi")
        assert r is None
        assert c is None
        # Feed "<thin" — still could be <think>
        r, c = parser.process_delta("<thin")
        assert r is None
        assert c is None
        # Feed "<think" — still could be <think>
        r, c = parser.process_delta("<think")
        assert r is None
        assert c is None
        # Feed "<think>" — now we're in think mode
        r, c = parser.process_delta("<think>")
        assert r is None  # No reasoning content yet
        assert c is None
        # Feed short reasoning — buffered to avoid partial </think>
        r, c = parser.process_delta("<think>Hello")
        assert r is None  # "Hello" (5 chars) < buffer threshold (7 chars)
        assert c is None
        # Feed enough reasoning to start emitting
        r, c = parser.process_delta("<think>Hello, world! This is reasoning.")
        assert r is not None  # Now there's enough to emit
        assert c is None

    def test_detect_state_not_think(self):
        """Text that starts with '<' but isn't '<think>'."""
        # "<tool" is not a prefix of "<think>", so it's flushed as content immediately
        parser = ReasoningParser()
        r, c = parser.process_delta("<tool")
        assert r is None
        assert c == "<tool"

        # "<to" is also not a prefix of "<think>"
        parser2 = ReasoningParser()
        r, c = parser2.process_delta("<to")
        assert r is None
        assert c == "<to"

    def test_detect_state_valid_prefix(self):
        """'<t' is a valid prefix of '<think>', so it should buffer."""
        parser = ReasoningParser()
        r, c = parser.process_delta("<t")
        assert r is None
        assert c is None  # Still buffering, could be <think>


# ===== starts_in_think tests (Qwen3 Coder scenario) =====

class TestExtractReasoningStartsInThink:
    """Non-streaming tests where prompt already opened <think>."""

    def test_basic(self):
        # Model output: reasoning directly, then </think>, then content
        text = "I'm thinking about this.\n</think>\nHello world!"
        reasoning, content = ReasoningParser.extract_reasoning(text, starts_in_think=True)
        assert reasoning == "I'm thinking about this."
        assert content == "Hello world!"

    def test_empty_reasoning(self):
        text = "</think>Content here."
        reasoning, content = ReasoningParser.extract_reasoning(text, starts_in_think=True)
        assert reasoning == ""
        assert content == "Content here."

    def test_no_close_tag(self):
        # No </think> found — model skipped reasoning, treat as content
        text = "Just reasoning forever..."
        reasoning, content = ReasoningParser.extract_reasoning(text, starts_in_think=True)
        assert reasoning is None
        assert content == text

    def test_no_close_tag_with_tool_calls(self):
        # Real-world case: model goes straight to tool calls without </think>
        text = 'Let me continue reading the file:\n\n<tool_call>\n<function=Bash>\n</tool_call>'
        reasoning, content = ReasoningParser.extract_reasoning(text, starts_in_think=True)
        assert reasoning is None
        assert content == text

    def test_redundant_think_tag(self):
        # Model redundantly outputs <think> when prompt already opened one
        text = "<think>\nActual reasoning\n</think>\nContent here."
        reasoning, content = ReasoningParser.extract_reasoning(text, starts_in_think=True)
        assert reasoning == "Actual reasoning"
        assert content == "Content here."

    def test_redundant_think_empty_reasoning(self):
        # Model outputs <think></think> redundantly
        text = "<think></think>Content here."
        reasoning, content = ReasoningParser.extract_reasoning(text, starts_in_think=True)
        assert reasoning == ""
        assert content == "Content here."

    def test_redundant_think_no_close(self):
        # Model outputs redundant <think> but no </think>
        text = "<think>\nSome content without close"
        reasoning, content = ReasoningParser.extract_reasoning(text, starts_in_think=True)
        assert reasoning is None
        assert content == text

    def test_multiline_reasoning(self):
        text = "Step 1: think\nStep 2: think more\n</think>\nResult."
        reasoning, content = ReasoningParser.extract_reasoning(text, starts_in_think=True)
        assert reasoning == "Step 1: think\nStep 2: think more"
        assert content == "Result."

    def test_newline_after_close(self):
        text = "Reasoning.\n</think>\nContent."
        reasoning, content = ReasoningParser.extract_reasoning(text, starts_in_think=True)
        assert reasoning == "Reasoning."
        assert content == "Content."

    def test_no_content_after_close(self):
        text = "Only reasoning.</think>"
        reasoning, content = ReasoningParser.extract_reasoning(text, starts_in_think=True)
        assert reasoning == "Only reasoning."
        assert content == ""

    def test_double_newline_after_close(self):
        """Real-world pattern: model outputs \\n before </think> and \\n\\n after."""
        text = "Reasoning here.\n</think>\n\nActual content."
        reasoning, content = ReasoningParser.extract_reasoning(text, starts_in_think=True)
        assert reasoning == "Reasoning here."
        assert content == "Actual content."

    def test_double_newline_non_starts_in_think(self):
        text = "<think>\nReasoning here.\n</think>\n\nActual content."
        reasoning, content = ReasoningParser.extract_reasoning(text)
        assert reasoning == "Reasoning here."
        assert content == "Actual content."


class TestStreamingStartsInThink:
    """Streaming tests where prompt already opened <think>."""

    def _feed_incremental(self, full_text: str, chunk_size: int = 3):
        parser = ReasoningParser(starts_in_think=True)
        reasoning_parts = []
        content_parts = []
        for i in range(0, len(full_text), chunk_size):
            chunk_text = full_text[:i + chunk_size]
            r_delta, c_delta = parser.process_delta(chunk_text)
            if r_delta:
                reasoning_parts.append(r_delta)
            if c_delta:
                content_parts.append(c_delta)
        return "".join(reasoning_parts), "".join(content_parts), parser

    def test_basic_streaming(self):
        text = "I need to think carefully.</think>The answer is 42."
        reasoning, content, parser = self._feed_incremental(text, chunk_size=5)
        assert reasoning == "I need to think carefully."
        assert content == "The answer is 42."
        assert parser.has_reasoning

    def test_char_by_char(self):
        text = "Reason.</think>Content."
        reasoning, content, parser = self._feed_incremental(text, chunk_size=1)
        assert reasoning == "Reason."
        assert content == "Content."

    def test_large_chunks(self):
        text = "Reasoning block.\n</think>Content block."
        reasoning, content, parser = self._feed_incremental(text, chunk_size=100)
        assert reasoning == "Reasoning block."
        assert content == "Content block."

    def test_empty_reasoning(self):
        text = "</think>Only content."
        reasoning, content, _ = self._feed_incremental(text, chunk_size=4)
        assert content == "Only content."

    def test_newline_after_close(self):
        text = "Think.\n</think>\nContent."
        reasoning, content, _ = self._feed_incremental(text, chunk_size=5)
        # Streaming deltas include the \n before </think> as it's emitted incrementally
        assert reasoning == "Think.\n"
        assert content == "Content."

    def test_get_reasoning_text_strips_newlines(self):
        """get_reasoning_text should strip leading/trailing newlines."""
        text = "Think.\n</think>\nContent."
        parser = ReasoningParser(starts_in_think=True)
        parser.process_delta(text)
        assert parser.get_reasoning_text(text) == "Think."

    def test_no_close_tag(self):
        """If model never outputs </think>, everything is reasoning."""
        text = "Just keeps thinking and thinking..."
        reasoning, content, parser = self._feed_incremental(text, chunk_size=5)
        # Everything is reasoning (minus buffered tail)
        assert len(reasoning) > 0
        assert content == ""
        assert parser.get_reasoning_text(text) == text

    def test_get_content_text(self):
        text = "Reasoning.</think>Content here."
        parser = ReasoningParser(starts_in_think=True)
        parser.process_delta(text)
        assert parser.get_content_text(text) == "Content here."
        assert parser.get_reasoning_text(text) == "Reasoning."

    def test_get_content_text_no_close(self):
        text = "Still thinking..."
        parser = ReasoningParser(starts_in_think=True)
        parser.process_delta(text)
        assert parser.get_content_text(text) == ""
        assert parser.get_reasoning_text(text) == "Still thinking..."

    def test_redundant_think_tag_streaming(self):
        """Model outputs redundant <think> when starts_in_think=True."""
        text = "<think>\nActual reasoning\n</think>\nContent here."
        reasoning, content, parser = self._feed_incremental(text, chunk_size=5)
        assert "<think>" not in reasoning
        assert "Actual reasoning" in reasoning
        assert content == "Content here."

    def test_redundant_think_tag_char_by_char(self):
        text = "<think>\nReason.\n</think>\nContent."
        reasoning, content, parser = self._feed_incremental(text, chunk_size=1)
        assert "<think>" not in reasoning
        assert content == "Content."
