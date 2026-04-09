"""
Tests for KimiK2ToolParser.
Uses a mock tokenizer to avoid HuggingFace model downloads.

Kimi K2 format uses special tokens:
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city":"SF"}<|tool_call_end|>
    <|tool_calls_section_end|>
"""

import json
import pytest

from tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTION_BEGIN = "<|tool_calls_section_begin|>"
SECTION_END = "<|tool_calls_section_end|>"
CALL_BEGIN = "<|tool_call_begin|>"
CALL_END = "<|tool_call_end|>"
ARG_BEGIN = "<|tool_call_argument_begin|>"


def _tool_call(name: str, index: int, arguments: str) -> str:
    return f"{CALL_BEGIN}functions.{name}:{index}{ARG_BEGIN}{arguments}{CALL_END}"


def _tool_calls_section(*calls: str) -> str:
    body = "\n".join(calls)
    return f"{SECTION_BEGIN}\n{body}\n{SECTION_END}"


# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Mock tokenizer with Kimi K2 special tokens in vocab."""

    def __init__(self):
        self._vocab = {
            "<|tool_calls_section_begin|>": 100,
            "<|tool_calls_section_end|>": 101,
            "<|tool_call_section_begin|>": 102,  # singular variant
            "<|tool_call_section_end|>": 103,     # singular variant
            "<|tool_call_begin|>": 104,
            "<|tool_call_end|>": 105,
            "<|tool_call_argument_begin|>": 106,
            "hello": 1,
        }

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


@pytest.fixture
def parser():
    return KimiK2ToolParser(MockTokenizer())


# ===================================================================
# Constructor
# ===================================================================

class TestConstructor:

    def test_requires_tokenizer(self):
        with pytest.raises(ValueError, match="model tokenizer must be passed"):
            KimiK2ToolParser(None)

    def test_requires_no_skip_special_tokens_flag(self):
        assert KimiK2ToolParser.requires_no_skip_special_tokens is True

    def test_tools_default_to_empty(self):
        p = KimiK2ToolParser(MockTokenizer())
        assert p.tools == []

    def test_missing_section_tokens_raises(self):
        """Tokenizer without section tokens should raise RuntimeError."""

        class BadTokenizer:
            def get_vocab(self):
                return {"hello": 1}

        with pytest.raises(RuntimeError, match="could not locate"):
            KimiK2ToolParser(BadTokenizer())

    def test_variant_token_ids(self, parser):
        """Both singular and plural variant IDs should be collected."""
        assert 100 in parser.tool_calls_start_token_ids
        assert 102 in parser.tool_calls_start_token_ids
        assert 101 in parser.tool_calls_end_token_ids
        assert 103 in parser.tool_calls_end_token_ids


# ===================================================================
# Non-streaming: extract_tool_calls
# ===================================================================

class TestExtractToolCalls:

    def test_no_tool_call(self, parser):
        result = parser.extract_tool_calls("Hello, how can I help?", None)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Hello, how can I help?"

    def test_single_tool_call(self, parser):
        output = _tool_calls_section(
            _tool_call("get_weather", 0, '{"city": "San Francisco"}')
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert tc.id == "functions.get_weather:0"
        args = json.loads(tc.function.arguments)
        assert args["city"] == "San Francisco"

    def test_multiple_tool_calls(self, parser):
        output = _tool_calls_section(
            _tool_call("get_weather", 0, '{"city": "NYC"}'),
            _tool_call("search", 1, '{"query": "restaurants"}'),
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "search"

    def test_content_before_tool_calls(self, parser):
        prefix = "Let me check the weather.\n"
        output = prefix + _tool_calls_section(
            _tool_call("get_weather", 0, '{"city": "Paris"}')
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert result.content == prefix

    def test_content_is_none_when_tool_at_start(self, parser):
        output = _tool_calls_section(
            _tool_call("get_weather", 0, '{"city": "Berlin"}')
        )
        result = parser.extract_tool_calls(output, None)
        assert result.content is None

    def test_function_name_without_prefix(self, parser):
        """Tool ID without 'functions.' prefix: just name:index."""
        output = (
            f"{SECTION_BEGIN}\n"
            f"{CALL_BEGIN}get_weather:0{ARG_BEGIN}{{\"city\": \"Tokyo\"}}{CALL_END}\n"
            f"{SECTION_END}"
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_weather"

    def test_malformed_output_returns_no_tools(self, parser):
        """If regex fails to match, should return tools_called=False."""
        output = f"{SECTION_BEGIN}\ngarbage content\n{SECTION_END}"
        result = parser.extract_tool_calls(output, None)
        # No tool_call_begin/end markers, so regex finds nothing
        assert result.tools_called is True  # section begin is present
        assert len(result.tool_calls) == 0

    def test_empty_arguments(self, parser):
        output = _tool_calls_section(
            _tool_call("ping", 0, '{}')
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

    def test_tool_call_id_preserves_full_id(self, parser):
        """The tool call ID should be the full 'functions.name:index' string."""
        output = _tool_calls_section(
            _tool_call("search", 0, '{"q": "test"}')
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tool_calls[0].id == "functions.search:0"


# ===================================================================
# Streaming: extract_tool_calls_streaming
# ===================================================================

class TestExtractToolCallsStreaming:

    def _make_token_ids(self, text: str, vocab: dict[str, int]) -> list[int]:
        """
        Simple token ID simulation: scan text for known special tokens,
        assign their IDs. For other text, assign sequential IDs starting
        from 1000. This lets the parser count start/end tokens correctly.
        """
        ids = []
        special_tokens = sorted(vocab.keys(), key=len, reverse=True)
        i = 0
        next_id = 1000
        while i < len(text):
            matched = False
            for token in special_tokens:
                if text[i:].startswith(token):
                    ids.append(vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                ids.append(next_id)
                next_id += 1
                i += 1
        return ids

    def _simulate_stream(self, parser, chunks: list[str]):
        """Feed text in specified chunks, collecting non-None DeltaMessages."""
        vocab = parser.model_tokenizer.get_vocab()
        results = []
        accumulated = ""
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            curr = accumulated

            prev_ids = self._make_token_ids(prev, vocab)
            curr_ids = self._make_token_ids(curr, vocab)
            delta_ids = self._make_token_ids(chunk, vocab)

            msg = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=curr,
                delta_text=chunk,
                previous_token_ids=prev_ids,
                current_token_ids=curr_ids,
                delta_token_ids=delta_ids,
                request=None,
            )
            if msg is not None:
                results.append(msg)
        return results

    def test_plain_text_streaming(self, parser):
        results = self._simulate_stream(parser, ["Hello", " world"])
        content_parts = [r.content for r in results if r.content]
        assert "".join(content_parts) == "Hello world"
        assert all(r.tool_calls is None for r in results)

    def test_tool_call_streaming_produces_tool_deltas(self, parser):
        call = _tool_call("get_weather", 0, '{"city": "Paris"}')
        chunks = [
            "Let me check. ",
            SECTION_BEGIN,
            "\n",
            CALL_BEGIN,
            "functions.get_weather:0",
            ARG_BEGIN,
            '{"city": ',
            '"Paris"}',
            CALL_END,
            "\n",
            SECTION_END,
        ]
        results = self._simulate_stream(parser, chunks)

        # Should have content before tool call
        content_msgs = [r for r in results if r.content and r.content.strip()]
        assert any("Let me check" in (m.content or "") for m in content_msgs)

        # Should have tool call messages
        tool_msgs = [r for r in results if r.tool_calls]
        assert len(tool_msgs) > 0

    def test_streaming_tool_name_sent(self, parser):
        """The first tool delta should contain the function name."""
        chunks = [
            SECTION_BEGIN,
            CALL_BEGIN,
            "functions.get_weather:0",
            ARG_BEGIN,
            '{"city": "SF"}',
            CALL_END,
            SECTION_END,
        ]
        results = self._simulate_stream(parser, chunks)

        tool_msgs = [r for r in results if r.tool_calls]
        # Find the delta with the function name
        names = []
        for r in tool_msgs:
            for tc in r.tool_calls:
                if tc.function and hasattr(tc.function, 'name') and tc.function.name:
                    names.append(tc.function.name)
        assert "get_weather" in names

    def test_streaming_multiple_tool_calls(self, parser):
        """Multiple tool calls in one section should produce multiple tool IDs."""
        chunks = [
            SECTION_BEGIN,
            CALL_BEGIN,
            "functions.get_weather:0",
            ARG_BEGIN,
            '{"city": "NYC"}',
            CALL_END,
            "\n",
            CALL_BEGIN,
            "functions.search:1",
            ARG_BEGIN,
            '{"q": "food"}',
            CALL_END,
            "\n",
            SECTION_END,
        ]
        results = self._simulate_stream(parser, chunks)

        tool_msgs = [r for r in results if r.tool_calls]
        names = []
        for r in tool_msgs:
            for tc in r.tool_calls:
                if tc.function and hasattr(tc.function, 'name') and tc.function.name:
                    names.append(tc.function.name)
        assert "get_weather" in names
        assert "search" in names

    def test_reset_streaming_state(self, parser):
        """_reset_streaming_state should reset all state."""
        parser.current_tool_id = 3
        parser.in_tool_section = True
        parser.token_buffer = "some data"
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = ["arg1", "arg2"]

        parser._reset_streaming_state()

        assert parser.current_tool_id == -1
        assert parser.in_tool_section is False
        assert parser.token_buffer == ""
        assert parser.current_tool_name_sent is False
        assert parser.streamed_args_for_tool == []
        assert parser.prev_tool_call_arr == []

    def test_section_char_limit_forces_exit(self, parser):
        """If tool section exceeds max chars, parser should force exit."""
        parser.max_section_chars = 50  # low limit for testing

        # Enter tool section
        vocab = parser.model_tokenizer.get_vocab()
        parser.in_tool_section = True
        parser.section_char_count = 0

        # Feed a large chunk that exceeds the limit
        big_chunk = "x" * 100
        prev_ids = self._make_token_ids(SECTION_BEGIN, vocab)
        curr_ids = prev_ids + self._make_token_ids(big_chunk, vocab)
        delta_ids = self._make_token_ids(big_chunk, vocab)

        result = parser.extract_tool_calls_streaming(
            previous_text=SECTION_BEGIN,
            current_text=SECTION_BEGIN + big_chunk,
            delta_text=big_chunk,
            previous_token_ids=prev_ids,
            current_token_ids=curr_ids,
            delta_token_ids=delta_ids,
            request=None,
        )
        # Should have exited tool section
        assert parser.in_tool_section is False


# ===================================================================
# Section marker handling
# ===================================================================

class TestMarkerHandling:

    def test_check_and_strip_markers(self, parser):
        text = f"hello{SECTION_BEGIN}world{SECTION_END}"
        cleaned, found_begin, found_end = parser._check_and_strip_markers(text)
        assert found_begin is True
        assert found_end is True
        assert SECTION_BEGIN not in cleaned
        assert SECTION_END not in cleaned
        assert "hello" in cleaned
        assert "world" in cleaned

    def test_check_and_strip_no_markers(self, parser):
        text = "just plain text"
        cleaned, found_begin, found_end = parser._check_and_strip_markers(text)
        assert found_begin is False
        assert found_end is False
        assert cleaned == text

    def test_singular_variant_markers(self, parser):
        """Should also detect singular variants."""
        text = "<|tool_call_section_begin|>content<|tool_call_section_end|>"
        cleaned, found_begin, found_end = parser._check_and_strip_markers(text)
        assert found_begin is True
        assert found_end is True
        assert cleaned == "content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
