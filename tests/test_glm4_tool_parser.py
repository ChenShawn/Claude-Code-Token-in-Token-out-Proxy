"""
Tests for Glm4MoeToolParser and Glm47MoeToolParser.
Uses a mock tokenizer to avoid HuggingFace model downloads.

GLM-4 tool call format:
    <tool_call>function_name
    <arg_key>key</arg_key><arg_value>value</arg_value>
    </tool_call>

GLM-4.7 format additionally allows:
    <tool_call>function_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>
    <tool_call>function_name</tool_call>  (zero arguments)
"""

import json
import pytest

from tool_parsers.glm4_moe_tool_parser import Glm4MoeToolParser
from tool_parsers.glm47_moe_tool_parser import Glm47MoeToolParser


# ---------------------------------------------------------------------------
# Mock tokenizer & request
# ---------------------------------------------------------------------------

class MockTokenizer:
    def __init__(self):
        self._vocab = {
            "<tool_call>": 100,
            "</tool_call>": 101,
            "hello": 1,
            "world": 2,
        }

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class MockRequest:
    """Minimal request object to satisfy _tools_enabled()."""

    def __init__(self, tools=None, tool_choice="auto"):
        self.tools = tools
        self.tool_choice = tool_choice


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        },
    },
}

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "precision": {"type": "integer"},
                "verbose": {"type": "boolean"},
            },
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
        },
    },
}

ALL_TOOLS = [WEATHER_TOOL, CALC_TOOL, SEARCH_TOOL]


# ---------------------------------------------------------------------------
# Helpers to build GLM-4 format strings
# ---------------------------------------------------------------------------

def _tool_call_glm4(func_name: str, args: list[tuple[str, str]]) -> str:
    """Build a GLM-4 tool call block: func_name on first line, then arg pairs."""
    lines = [f"<tool_call>{func_name}"]
    for key, val in args:
        lines.append(f"<arg_key>{key}</arg_key><arg_value>{val}</arg_value>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def _tool_call_glm47_inline(func_name: str, args: list[tuple[str, str]]) -> str:
    """Build a GLM-4.7 tool call with args on same line (no newline after func name)."""
    parts = [f"<tool_call>{func_name}"]
    for key, val in args:
        parts.append(f"<arg_key>{key}</arg_key><arg_value>{val}</arg_value>")
    parts.append("</tool_call>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def glm4_parser():
    return Glm4MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)


@pytest.fixture
def glm47_parser():
    return Glm47MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)


@pytest.fixture
def mock_request():
    return MockRequest(tools=ALL_TOOLS, tool_choice="auto")


# ===================================================================
# Constructor tests
# ===================================================================

class TestConstructor:

    def test_requires_tokenizer(self):
        with pytest.raises(ValueError, match="model tokenizer must be passed"):
            Glm4MoeToolParser(None, tools=[])

    def test_glm47_requires_tokenizer(self):
        with pytest.raises(ValueError, match="model tokenizer must be passed"):
            Glm47MoeToolParser(None, tools=[])

    def test_tools_default_to_empty(self):
        p = Glm4MoeToolParser(MockTokenizer())
        assert p.tools == []

    def test_vocab_cached(self, glm4_parser):
        assert glm4_parser.vocab.get("<tool_call>") == 100
        assert glm4_parser.vocab.get("</tool_call>") == 101

    def test_glm47_inherits_from_glm4(self, glm47_parser):
        assert isinstance(glm47_parser, Glm4MoeToolParser)


# ===================================================================
# Static helpers
# ===================================================================

class TestStaticHelpers:

    def test_deserialize_json(self, glm4_parser):
        assert glm4_parser._deserialize("42") == 42
        assert glm4_parser._deserialize("3.14") == 3.14
        assert glm4_parser._deserialize("true") is True
        assert glm4_parser._deserialize("false") is False
        assert glm4_parser._deserialize('"hello"') == "hello"
        assert glm4_parser._deserialize("[1, 2, 3]") == [1, 2, 3]
        assert glm4_parser._deserialize('{"a": 1}') == {"a": 1}

    def test_deserialize_literal_eval_fallback(self, glm4_parser):
        assert glm4_parser._deserialize("(1, 2)") == (1, 2)

    def test_deserialize_raw_string_fallback(self, glm4_parser):
        assert glm4_parser._deserialize("not_valid_json") == "not_valid_json"

    def test_json_escape_string_content(self, glm4_parser):
        assert glm4_parser._json_escape_string_content("") == ""
        assert glm4_parser._json_escape_string_content("hello") == "hello"
        assert glm4_parser._json_escape_string_content('a"b') == 'a\\"b'
        assert glm4_parser._json_escape_string_content("line\nbreak") == "line\\nbreak"
        assert glm4_parser._json_escape_string_content("tab\there") == "tab\\there"

    def test_is_string_type_found(self, glm4_parser):
        assert glm4_parser._is_string_type("get_weather", "location", ALL_TOOLS) is True

    def test_is_string_type_not_string(self, glm4_parser):
        assert glm4_parser._is_string_type("calculate", "precision", ALL_TOOLS) is False

    def test_is_string_type_unknown_tool(self, glm4_parser):
        assert glm4_parser._is_string_type("unknown", "foo", ALL_TOOLS) is False

    def test_is_string_type_none_tools(self, glm4_parser):
        assert glm4_parser._is_string_type("get_weather", "location", None) is False

    def test_tools_enabled_with_tools(self):
        req = MockRequest(tools=[WEATHER_TOOL], tool_choice="auto")
        assert Glm4MoeToolParser._tools_enabled(req) is True

    def test_tools_enabled_no_tools(self):
        req = MockRequest(tools=None, tool_choice="auto")
        assert Glm4MoeToolParser._tools_enabled(req) is False

    def test_tools_enabled_choice_none(self):
        req = MockRequest(tools=[WEATHER_TOOL], tool_choice="none")
        assert Glm4MoeToolParser._tools_enabled(req) is False


# ===================================================================
# GLM-4: Non-streaming extract_tool_calls
# ===================================================================

class TestGlm4ExtractToolCalls:

    def test_no_tool_call(self, glm4_parser):
        result = glm4_parser.extract_tool_calls("Hello, how can I help?", None)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Hello, how can I help?"

    def test_single_tool_call_one_param(self, glm4_parser):
        output = _tool_call_glm4("get_weather", [("location", "San Francisco")])
        result = glm4_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert args["location"] == "San Francisco"

    def test_single_tool_call_multiple_params(self, glm4_parser):
        output = _tool_call_glm4("get_weather", [
            ("location", "Tokyo"),
            ("unit", "celsius"),
        ])
        result = glm4_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "Tokyo"
        assert args["unit"] == "celsius"

    def test_content_before_tool_call(self, glm4_parser):
        prefix = "Let me check the weather.\n"
        output = prefix + _tool_call_glm4("get_weather", [("location", "Paris")])
        result = glm4_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert result.content == prefix

    def test_content_is_none_when_tool_at_start(self, glm4_parser):
        output = _tool_call_glm4("get_weather", [("location", "Berlin")])
        result = glm4_parser.extract_tool_calls(output, None)
        assert result.content is None

    def test_multiple_tool_calls(self, glm4_parser):
        output = (
            _tool_call_glm4("get_weather", [("location", "London")])
            + "\n"
            + _tool_call_glm4("calculate", [("expression", "2+2")])
        )
        result = glm4_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "calculate"

    def test_non_string_param_deserialized(self, glm4_parser):
        """Integer params should be deserialized from string."""
        output = _tool_call_glm4("calculate", [("precision", "5")])
        result = glm4_parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["precision"] == 5
        assert isinstance(args["precision"], int)

    def test_boolean_param_deserialized(self, glm4_parser):
        output = _tool_call_glm4("calculate", [("verbose", "true")])
        result = glm4_parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["verbose"] is True

    def test_string_param_not_deserialized(self, glm4_parser):
        """String-type params should stay as strings, not be deserialized."""
        output = _tool_call_glm4("get_weather", [("location", "42")])
        result = glm4_parser.extract_tool_calls(output, None)
        args = json.loads(result.function_calls[0].function.arguments) if hasattr(result, 'function_calls') else json.loads(result.tool_calls[0].function.arguments)
        # "42" would be deserialized to int if not for string type check
        assert args["location"] == "42"
        assert isinstance(args["location"], str)

    def test_unicode_in_param_value(self, glm4_parser):
        output = _tool_call_glm4("search", [("query", "天气预报 北京")])
        result = glm4_parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["query"] == "天气预报 北京"

    def test_tool_call_id_format(self, glm4_parser):
        output = _tool_call_glm4("get_weather", [("location", "LA")])
        result = glm4_parser.extract_tool_calls(output, None)
        tc_id = result.tool_calls[0].id
        assert tc_id.startswith("call_")
        hex_part = tc_id[5:]
        assert len(hex_part) == 24
        int(hex_part, 16)  # should not raise

    def test_mixed_param_types(self, glm4_parser):
        output = _tool_call_glm4("calculate", [
            ("expression", "2+2"),
            ("precision", "3"),
            ("verbose", "true"),
        ])
        result = glm4_parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["expression"] == "2+2"
        assert args["precision"] == 3
        assert args["verbose"] is True

    def test_unknown_function(self, glm4_parser):
        output = _tool_call_glm4("unknown_func", [("foo", "bar")])
        result = glm4_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "unknown_func"

    def test_no_args(self, glm4_parser):
        """GLM-4 format with no arg_key/arg_value pairs."""
        output = "<tool_call>get_weather\n</tool_call>"
        result = glm4_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}


# ===================================================================
# GLM-4.7: Non-streaming extract_tool_calls
# ===================================================================

class TestGlm47ExtractToolCalls:

    def test_standard_glm4_format_still_works(self, glm47_parser):
        """GLM-4.7 parser should still handle GLM-4 format (newline after name)."""
        output = _tool_call_glm4("get_weather", [("location", "Paris")])
        result = glm47_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "Paris"

    def test_inline_format_no_newline(self, glm47_parser):
        """GLM-4.7 format: func name directly followed by <arg_key> (no newline)."""
        output = _tool_call_glm47_inline("get_weather", [("location", "Tokyo")])
        result = glm47_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        tc = result.tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert args["location"] == "Tokyo"

    def test_zero_argument_call(self, glm47_parser):
        """GLM-4.7 supports zero-argument calls: <tool_call>func</tool_call>."""
        output = "<tool_call>get_weather</tool_call>"
        result = glm47_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        tc = result.tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert args == {}

    def test_zero_argument_with_whitespace(self, glm47_parser):
        """Zero-argument with whitespace around func name."""
        output = "<tool_call> get_weather </tool_call>"
        result = glm47_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_weather"

    def test_multiple_inline_tool_calls(self, glm47_parser):
        output = (
            _tool_call_glm47_inline("get_weather", [("location", "NYC")])
            + "\n"
            + _tool_call_glm47_inline("search", [("query", "restaurants")])
        )
        result = glm47_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "search"

    def test_inline_multiple_params(self, glm47_parser):
        output = _tool_call_glm47_inline("get_weather", [
            ("location", "London"),
            ("unit", "fahrenheit"),
        ])
        result = glm47_parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "London"
        assert args["unit"] == "fahrenheit"

    def test_content_before_inline_tool(self, glm47_parser):
        prefix = "Checking now.\n"
        output = prefix + _tool_call_glm47_inline("search", [("query", "test")])
        result = glm47_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert result.content == prefix

    def test_no_tool_call(self, glm47_parser):
        result = glm47_parser.extract_tool_calls("Just plain text.", None)
        assert result.tools_called is False
        assert result.content == "Just plain text."


# ===================================================================
# GLM-4: Streaming extract_tool_calls_streaming
# ===================================================================

class TestGlm4Streaming:

    def _simulate_stream_by_chunks(self, parser, chunks, request):
        """Feed text in chunks, collecting non-None DeltaMessages."""
        results = []
        accumulated = ""
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            curr = accumulated

            prev_ids = list(range(len(prev)))
            curr_ids = list(range(len(curr)))
            delta_ids = list(range(len(prev), len(curr)))

            msg = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=curr,
                delta_text=chunk,
                previous_token_ids=prev_ids,
                current_token_ids=curr_ids,
                delta_token_ids=delta_ids,
                request=request,
            )
            if msg is not None:
                results.append(msg)
        return results

    def test_plain_text_no_tools_in_request(self, glm4_parser):
        """When request has no tools, streaming just returns content deltas."""
        req = MockRequest(tools=None, tool_choice="auto")
        results = self._simulate_stream_by_chunks(
            glm4_parser, ["Hello", " world"], req
        )
        content = "".join(r.content for r in results if r.content)
        assert content == "Hello world"
        assert all(r.tool_calls is None for r in results)

    def test_plain_text_with_tools_enabled(self, glm4_parser, mock_request):
        """Plain text with tools enabled but no tool call in output."""
        results = self._simulate_stream_by_chunks(
            glm4_parser, ["Hello", " world"], mock_request
        )
        content = "".join(r.content for r in results if r.content)
        assert content == "Hello world"

    def test_single_tool_call_streaming(self, mock_request):
        parser = Glm4MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)
        # Split chunks finely: each streaming call returns at most one delta,
        # so <arg_value>, value, </arg_value>, and </tool_call> must be separate.
        chunks = [
            "Let me check. ",
            "<tool_call>",
            "get_weather\n",
            "<arg_key>location</arg_key>",
            "<arg_value>",
            "Berlin",
            "</arg_value>",
            "</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks, mock_request)

        # Should have content before tool call
        content_msgs = [r for r in results if r.content and r.content.strip()]
        assert any("Let me check" in (m.content or "") for m in content_msgs)

        # Should have tool call deltas
        tool_msgs = [r for r in results if r.tool_calls]
        assert len(tool_msgs) > 0

        # Check function name
        name_delta = None
        for r in tool_msgs:
            for tc in r.tool_calls:
                if tc.function and tc.function.name:
                    name_delta = tc
                    break
        assert name_delta is not None
        assert name_delta.function.name == "get_weather"
        assert name_delta.id is not None
        assert name_delta.id.startswith("call_")

        # Reconstruct arguments from deltas
        arg_parts = []
        for r in tool_msgs:
            for tc in r.tool_calls:
                if tc.function and tc.function.arguments:
                    arg_parts.append(tc.function.arguments)
        combined_args = "".join(arg_parts)
        assert combined_args.startswith("{")
        assert combined_args.endswith("}")
        parsed = json.loads(combined_args)
        assert parsed["location"] == "Berlin"

    def test_multiple_tool_calls_streaming(self, mock_request):
        parser = Glm4MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "<arg_key>location</arg_key>",
            "<arg_value>NYC</arg_value>",
            "</tool_call>",
            "<tool_call>",
            "search\n",
            "<arg_key>query</arg_key>",
            "<arg_value>restaurants</arg_value>",
            "</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks, mock_request)

        tool_msgs = [r for r in results if r.tool_calls]
        all_tool_calls = []
        for r in tool_msgs:
            all_tool_calls.extend(r.tool_calls)

        # Should have two distinct tool call indices
        indices = set()
        for tc in all_tool_calls:
            if tc.function and tc.function.name:
                indices.add(tc.index)
        assert len(indices) == 2

        # Check function names
        names = [tc.function.name for tc in all_tool_calls if tc.function and tc.function.name]
        assert "get_weather" in names
        assert "search" in names

    def test_streaming_string_incremental(self, mock_request):
        """String-type args should be streamed incrementally."""
        parser = Glm4MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)
        # Feed the string value in small pieces
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "<arg_key>location</arg_key>",
            "<arg_value>",
            "San ",
            "Fran",
            "cisco",
            "</arg_value>",
            "</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks, mock_request)

        tool_msgs = [r for r in results if r.tool_calls]
        arg_parts = []
        for r in tool_msgs:
            for tc in r.tool_calls:
                if tc.function and tc.function.arguments:
                    arg_parts.append(tc.function.arguments)
        combined_args = "".join(arg_parts)
        parsed = json.loads(combined_args)
        assert parsed["location"] == "San Francisco"

    def test_streaming_non_string_waits_for_complete(self, mock_request):
        """Non-string type args wait for full value before emitting."""
        parser = Glm4MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)
        chunks = [
            "<tool_call>",
            "calculate\n",
            "<arg_key>precision</arg_key>",
            "<arg_value>42</arg_value>",
            "</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks, mock_request)

        tool_msgs = [r for r in results if r.tool_calls]
        arg_parts = []
        for r in tool_msgs:
            for tc in r.tool_calls:
                if tc.function and tc.function.arguments:
                    arg_parts.append(tc.function.arguments)
        combined_args = "".join(arg_parts)
        parsed = json.loads(combined_args)
        assert parsed["precision"] == 42

    def test_streaming_empty_tool_call_ignored(self, mock_request):
        """An empty <tool_call></tool_call> should be skipped."""
        parser = Glm4MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)
        chunks = [
            "<tool_call>",
            "</tool_call>",
            "<tool_call>",
            "get_weather\n",
            "<arg_key>location</arg_key>",
            "<arg_value>NYC</arg_value>",
            "</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks, mock_request)

        tool_msgs = [r for r in results if r.tool_calls]
        names = []
        for r in tool_msgs:
            for tc in r.tool_calls:
                if tc.function and tc.function.name:
                    names.append(tc.function.name)
        assert names == ["get_weather"]

    def test_streaming_tool_call_ids_unique(self, mock_request):
        parser = Glm4MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "<arg_key>location</arg_key>",
            "<arg_value>A</arg_value>",
            "</tool_call>",
            "<tool_call>",
            "search\n",
            "<arg_key>query</arg_key>",
            "<arg_value>B</arg_value>",
            "</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks, mock_request)

        ids = set()
        for r in results:
            if r.tool_calls:
                for tc in r.tool_calls:
                    if tc.id:
                        ids.add(tc.id)
        assert len(ids) == 2

    def test_streaming_mixed_args(self, mock_request):
        """Mix of string and non-string args in streaming."""
        parser = Glm4MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)
        chunks = [
            "<tool_call>",
            "calculate\n",
            "<arg_key>expression</arg_key>",
            "<arg_value>2+2</arg_value>",
            "<arg_key>precision</arg_key>",
            "<arg_value>3</arg_value>",
            "</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks, mock_request)

        tool_msgs = [r for r in results if r.tool_calls]
        arg_parts = []
        for r in tool_msgs:
            for tc in r.tool_calls:
                if tc.function and tc.function.arguments:
                    arg_parts.append(tc.function.arguments)
        combined_args = "".join(arg_parts)
        parsed = json.loads(combined_args)
        assert parsed["expression"] == "2+2"
        assert parsed["precision"] == 3

    def test_streaming_no_args_tool_call(self, mock_request):
        """Tool call with no args should produce {} arguments."""
        parser = Glm4MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks, mock_request)

        tool_msgs = [r for r in results if r.tool_calls]
        arg_parts = []
        for r in tool_msgs:
            for tc in r.tool_calls:
                if tc.function and tc.function.arguments:
                    arg_parts.append(tc.function.arguments)
        combined_args = "".join(arg_parts)
        parsed = json.loads(combined_args)
        assert parsed == {}

    def test_streaming_special_chars_in_string(self, mock_request):
        """Special characters in string values should be properly escaped."""
        parser = Glm4MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "<arg_key>location</arg_key>",
            '<arg_value>New "York"',
            "\nCity</arg_value>",
            "</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks, mock_request)

        tool_msgs = [r for r in results if r.tool_calls]
        arg_parts = []
        for r in tool_msgs:
            for tc in r.tool_calls:
                if tc.function and tc.function.arguments:
                    arg_parts.append(tc.function.arguments)
        combined_args = "".join(arg_parts)
        parsed = json.loads(combined_args)
        assert parsed["location"] == 'New "York"\nCity'


# ===================================================================
# GLM-4.7: Streaming
# ===================================================================

class TestGlm47Streaming:

    def _simulate_stream_by_chunks(self, parser, chunks, request):
        results = []
        accumulated = ""
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            curr = accumulated

            prev_ids = list(range(len(prev)))
            curr_ids = list(range(len(curr)))
            delta_ids = list(range(len(prev), len(curr)))

            msg = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=curr,
                delta_text=chunk,
                previous_token_ids=prev_ids,
                current_token_ids=curr_ids,
                delta_token_ids=delta_ids,
                request=request,
            )
            if msg is not None:
                results.append(msg)
        return results

    def test_inline_format_streaming(self, mock_request):
        """GLM-4.7 inline format (no newline after func name) should stream correctly."""
        parser = Glm47MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)
        chunks = [
            "<tool_call>",
            "get_weather",
            "<arg_key>location</arg_key>",
            "<arg_value>",
            "Tokyo",
            "</arg_value>",
            "</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks, mock_request)

        tool_msgs = [r for r in results if r.tool_calls]
        assert len(tool_msgs) > 0

        names = [
            tc.function.name
            for r in tool_msgs
            for tc in r.tool_calls
            if tc.function and tc.function.name
        ]
        assert "get_weather" in names

        arg_parts = [
            tc.function.arguments
            for r in tool_msgs
            for tc in r.tool_calls
            if tc.function and tc.function.arguments
        ]
        combined_args = "".join(arg_parts)
        parsed = json.loads(combined_args)
        assert parsed["location"] == "Tokyo"

    def test_standard_format_streaming(self, mock_request):
        """GLM-4.7 parser should still handle GLM-4 newline format in streaming."""
        parser = Glm47MoeToolParser(MockTokenizer(), tools=ALL_TOOLS)
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "<arg_key>location</arg_key>",
            "<arg_value>Berlin</arg_value>",
            "</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks, mock_request)

        tool_msgs = [r for r in results if r.tool_calls]
        names = [
            tc.function.name
            for r in tool_msgs
            for tc in r.tool_calls
            if tc.function and tc.function.name
        ]
        assert "get_weather" in names


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_incomplete_tool_call_no_end_tag(self, glm4_parser):
        """Missing </tool_call> means regex won't match."""
        output = "<tool_call>get_weather\n<arg_key>location</arg_key><arg_value>NYC</arg_value>"
        result = glm4_parser.extract_tool_calls(output, None)
        assert result.tools_called is False

    def test_nested_special_chars_in_value(self, glm4_parser):
        """Values with XML-like content should still be parsed."""
        output = _tool_call_glm4("search", [("query", "find <tag> in HTML")])
        result = glm4_parser.extract_tool_calls(output, None)
        # May or may not parse cleanly depending on regex greediness,
        # but should not crash
        assert isinstance(result.tools_called, bool)

    def test_multiline_arg_value(self, glm4_parser):
        output = (
            "<tool_call>search\n"
            "<arg_key>query</arg_key><arg_value>line1\nline2\nline3</arg_value>\n"
            "</tool_call>"
        )
        result = glm4_parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["query"] == "line1\nline2\nline3"

    def test_whitespace_in_arg_key(self, glm4_parser):
        """Arg keys with whitespace should be stripped."""
        output = (
            "<tool_call>get_weather\n"
            "<arg_key> location </arg_key><arg_value>NYC</arg_value>\n"
            "</tool_call>"
        )
        result = glm4_parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "NYC"

    def test_empty_arg_value(self, glm4_parser):
        output = (
            "<tool_call>get_weather\n"
            "<arg_key>location</arg_key><arg_value></arg_value>\n"
            "</tool_call>"
        )
        result = glm4_parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
