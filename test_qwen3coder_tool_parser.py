"""
Tests for Qwen3CoderToolParser.
Uses a mock tokenizer to avoid HuggingFace model downloads.
"""

import json
import pytest

from qwen3coder_tool_parser import Qwen3CoderToolParser
from tool_types import DeltaMessage


# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal tokenizer mock that satisfies Qwen3CoderToolParser requirements."""

    def __init__(self):
        self._vocab = {
            "<tool_call>": 100,
            "</tool_call>": 101,
            # Add some regular tokens
            "hello": 1,
            "world": 2,
        }

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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
                "scores": {"type": "array"},
                "filters": {"type": "object"},
            },
        },
    },
}

ALL_TOOLS = [WEATHER_TOOL, CALC_TOOL, SEARCH_TOOL]


@pytest.fixture
def parser():
    return Qwen3CoderToolParser(MockTokenizer(), tools=ALL_TOOLS)


# ===================================================================
# Non-streaming: extract_tool_calls
# ===================================================================

class TestExtractToolCalls:

    def test_no_tool_call(self, parser):
        """Plain text with no tool call markers."""
        result = parser.extract_tool_calls("Hello, how can I help?", None)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Hello, how can I help?"

    def test_single_tool_call_basic(self, parser):
        """Single tool call with one string parameter."""
        output = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>San Francisco</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert args["location"] == "San Francisco"

    def test_single_tool_call_multiple_params(self, parser):
        """Single tool call with multiple parameters."""
        output = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>Tokyo</parameter>\n"
            "<parameter=unit>celsius</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "Tokyo"
        assert args["unit"] == "celsius"

    def test_content_before_tool_call(self, parser):
        """Text content preceding the tool call should be captured."""
        output = (
            "Let me check the weather for you.\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>Paris</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert result.content == "Let me check the weather for you.\n"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "Paris"

    def test_multiple_tool_calls(self, parser):
        """Multiple tool calls in one output."""
        output = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>London</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=calculate>\n"
            "<parameter=expression>2+2</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "calculate"

    def test_no_parameters(self, parser):
        """Tool call with no parameters yields empty dict."""
        output = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

    def test_unknown_function_name(self, parser):
        """Function not in tools list still parses; params treated as strings."""
        output = (
            "<tool_call>\n"
            "<function=unknown_func>\n"
            "<parameter=foo>bar</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        tc = result.tool_calls[0]
        assert tc.function.name == "unknown_func"
        args = json.loads(tc.function.arguments)
        assert args["foo"] == "bar"


# ===================================================================
# Parameter type conversion
# ===================================================================

class TestParamConversion:

    def test_integer_conversion(self, parser):
        output = (
            "<tool_call>\n"
            "<function=calculate>\n"
            "<parameter=precision>5</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["precision"] == 5
        assert isinstance(args["precision"], int)

    def test_boolean_true(self, parser):
        output = (
            "<tool_call>\n"
            "<function=calculate>\n"
            "<parameter=verbose>true</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["verbose"] is True

    def test_boolean_false(self, parser):
        output = (
            "<tool_call>\n"
            "<function=calculate>\n"
            "<parameter=verbose>false</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["verbose"] is False

    def test_boolean_invalid_degenerates_to_false(self, parser):
        output = (
            "<tool_call>\n"
            "<function=calculate>\n"
            "<parameter=verbose>maybe</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["verbose"] is False

    def test_null_value(self, parser):
        output = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>null</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] is None

    def test_array_param(self, parser):
        output = (
            "<tool_call>\n"
            "<function=search>\n"
            '<parameter=scores>[0.9, 0.8, 0.7]</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["scores"] == [0.9, 0.8, 0.7]

    def test_object_param(self, parser):
        output = (
            "<tool_call>\n"
            "<function=search>\n"
            '<parameter=filters>{"category": "news"}</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["filters"] == {"category": "news"}

    def test_integer_invalid_returns_string(self, parser):
        output = (
            "<tool_call>\n"
            "<function=calculate>\n"
            "<parameter=precision>not_a_number</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["precision"] == "not_a_number"

    def test_float_conversion(self, parser):
        """Float type with decimal value stays float; integer-valued float becomes int."""
        tool = {
            "type": "function",
            "function": {
                "name": "math_op",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "val": {"type": "number"},
                    },
                },
            },
        }
        p = Qwen3CoderToolParser(MockTokenizer(), tools=[tool])
        output = (
            "<tool_call>\n"
            "<function=math_op>\n"
            "<parameter=val>3.14</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = p.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["val"] == 3.14

        output2 = (
            "<tool_call>\n"
            "<function=math_op>\n"
            "<parameter=val>5.0</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result2 = p.extract_tool_calls(output2, None)
        args2 = json.loads(result2.tool_calls[0].function.arguments)
        # 5.0 has no fractional part, so it becomes int 5
        assert args2["val"] == 5
        assert isinstance(args2["val"], int)


# ===================================================================
# _get_arguments_config
# ===================================================================

class TestGetArgumentsConfig:

    def test_finds_matching_tool(self, parser):
        config = parser._get_arguments_config("get_weather", ALL_TOOLS)
        assert "location" in config
        assert "unit" in config

    def test_no_match_returns_empty(self, parser):
        config = parser._get_arguments_config("nonexistent", ALL_TOOLS)
        assert config == {}

    def test_none_tools_returns_empty(self, parser):
        config = parser._get_arguments_config("get_weather", None)
        assert config == {}

    def test_non_function_type_skipped(self, parser):
        tools = [{"type": "retrieval", "function": {"name": "get_weather"}}]
        config = parser._get_arguments_config("get_weather", tools)
        assert config == {}


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_param_with_leading_trailing_newlines(self, parser):
        """Newlines at start/end of parameter value should be stripped."""
        output = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>\nNew York\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "New York"

    def test_param_value_with_special_chars(self, parser):
        """Parameter values can contain special characters."""
        output = (
            "<tool_call>\n"
            "<function=search>\n"
            '<parameter=query>what is "AI" & how does it work?</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["query"] == 'what is "AI" & how does it work?'

    def test_multiline_param_value(self, parser):
        """Parameter value spanning multiple lines."""
        output = (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>\nline one\nline two\nline three\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "line one\nline two\nline three" == args["query"]

    def test_tool_call_id_format(self, parser):
        """Tool call IDs should start with 'call_' and have 24 hex chars."""
        output = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>LA</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(output, None)
        tc_id = result.tool_calls[0].id
        assert tc_id.startswith("call_")
        hex_part = tc_id[5:]
        assert len(hex_part) == 24
        int(hex_part, 16)  # should not raise

    def test_constructor_requires_tokenizer(self):
        """Passing None tokenizer should raise ValueError."""
        with pytest.raises(ValueError, match="model tokenizer must be passed"):
            Qwen3CoderToolParser(None, tools=[])

    def test_constructor_requires_special_tokens(self):
        """Tokenizer without tool_call tokens should raise RuntimeError."""

        class BadTokenizer:
            def get_vocab(self):
                return {"hello": 1}

        with pytest.raises(RuntimeError, match="could not locate tool call"):
            Qwen3CoderToolParser(BadTokenizer(), tools=[])


# ===================================================================
# Streaming: extract_tool_calls_streaming
# ===================================================================

class TestExtractToolCallsStreaming:
    """
    Simulate streaming by feeding characters incrementally.
    """

    def _simulate_stream(self, parser, full_text: str, token_ids=None):
        """
        Feed full_text one character at a time, collecting non-None DeltaMessages.
        Returns list of DeltaMessage.
        """
        results = []
        for i in range(1, len(full_text) + 1):
            prev = full_text[: i - 1]
            curr = full_text[:i]
            delta = full_text[i - 1 : i]

            # Assign special token IDs for sentinel tokens
            delta_ids = []
            if delta and token_ids:
                delta_ids = token_ids.get(i - 1, [])
            elif curr.endswith("<tool_call>"):
                delta_ids = [100]
            elif curr.endswith("</tool_call>"):
                delta_ids = [101]

            prev_ids = list(range(i - 1))
            curr_ids = list(range(i))

            msg = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=curr,
                delta_text=delta,
                previous_token_ids=prev_ids,
                current_token_ids=curr_ids,
                delta_token_ids=delta_ids,
                request=None,
            )
            if msg is not None:
                results.append(msg)
        return results

    def _simulate_stream_by_chunks(self, parser, chunks: list[str]):
        """
        Feed text in specified chunks (more realistic than char-by-char).
        Returns list of non-None DeltaMessages.
        """
        results = []
        accumulated = ""
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            curr = accumulated

            delta_ids = []
            if curr.endswith("<tool_call>"):
                delta_ids = [100]
            elif curr.endswith("</tool_call>"):
                delta_ids = [101]

            prev_ids = list(range(len(prev)))
            curr_ids = list(range(len(curr)))

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
        """Plain text without tool calls streams content deltas."""
        results = self._simulate_stream(parser, "Hello world")
        content_parts = [r.content for r in results if r.content is not None]
        assert "".join(content_parts) == "Hello world"
        # No tool calls
        assert all(r.tool_calls is None for r in results)

    def test_single_tool_streaming_chunks(self, parser):
        """Streaming a single tool call in realistic chunks."""
        chunks = [
            "Let me check. ",
            "<tool_call>",
            "\n<function=",
            "get_weather>",
            "\n<parameter=location>",
            "Berlin",
            "</parameter>",
            "\n</function>",
            "\n</tool_call>",
        ]
        results = self._simulate_stream_by_chunks(parser, chunks)

        # Should have content before tool call
        content_msgs = [r for r in results if r.content is not None]
        assert len(content_msgs) > 0

        # Should have tool call messages
        tool_msgs = [r for r in results if r.tool_calls is not None]
        assert len(tool_msgs) > 0

        # Check function name was sent
        name_msgs = [
            r
            for r in tool_msgs
            if r.tool_calls[0].function and r.tool_calls[0].function.name
        ]
        assert len(name_msgs) == 1
        assert name_msgs[0].tool_calls[0].function.name == "get_weather"

        # Check arguments were streamed (should include "{", param, "}")
        arg_parts = []
        for r in tool_msgs:
            if r.tool_calls[0].function and r.tool_calls[0].function.arguments:
                arg_parts.append(r.tool_calls[0].function.arguments)
        combined_args = "".join(arg_parts)
        assert combined_args.startswith("{")
        assert combined_args.endswith("}")
        parsed = json.loads(combined_args)
        assert parsed["location"] == "Berlin"

    def test_streaming_resets_on_empty_previous(self, parser):
        """First call with empty previous_text resets state."""
        # First call
        parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="H",
            delta_text="H",
            previous_token_ids=[],
            current_token_ids=[1],
            delta_token_ids=[1],
            request=None,
        )
        assert parser.accumulated_text == "H"

        # Second "first" call resets
        parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="X",
            delta_text="X",
            previous_token_ids=[],
            current_token_ids=[2],
            delta_token_ids=[2],
            request=None,
        )
        assert parser.accumulated_text == "X"

    def test_streaming_empty_delta_returns_none(self, parser):
        """Empty delta_text with no special token IDs returns None."""
        result = parser.extract_tool_calls_streaming(
            previous_text="hello",
            current_text="hello",
            delta_text="",
            previous_token_ids=[1],
            current_token_ids=[1],
            delta_token_ids=[],
            request=None,
        )
        assert result is None


# ===================================================================
# _parse_xml_function_call directly
# ===================================================================

class TestParseXmlFunctionCall:

    def test_basic_parse(self, parser):
        func_str = "get_weather>\n<parameter=location>NYC</parameter>\n"
        tc = parser._parse_xml_function_call(func_str, ALL_TOOLS)
        assert tc is not None
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert args["location"] == "NYC"

    def test_no_closing_angle_bracket(self, parser):
        """If function name has no '>' delimiter, returns None."""
        tc = parser._parse_xml_function_call("get_weather", ALL_TOOLS)
        assert tc is None

    def test_params_without_closing_tags(self, parser):
        """Parameters terminated by next parameter or function end."""
        func_str = (
            "get_weather>\n"
            "<parameter=location>Boston<parameter=unit>fahrenheit</function>"
        )
        tc = parser._parse_xml_function_call(func_str, ALL_TOOLS)
        assert tc is not None
        args = json.loads(tc.function.arguments)
        assert args["location"] == "Boston"
        assert args["unit"] == "fahrenheit"


# ===================================================================
# _get_function_calls
# ===================================================================

class TestGetFunctionCalls:

    def test_extracts_from_complete_tool_call(self, parser):
        text = (
            "<tool_call>\n"
            "<function=foo>bar</function>\n"
            "</tool_call>"
        )
        calls = parser._get_function_calls(text)
        assert len(calls) >= 1
        assert "foo>" in calls[0] or calls[0].startswith("foo>")

    def test_extracts_from_incomplete_tool_call(self, parser):
        """Incomplete tool_call (no </tool_call>) should still match."""
        text = "<tool_call>\n<function=bar>baz</function>\n"
        calls = parser._get_function_calls(text)
        assert len(calls) >= 1

    def test_multiple_function_calls_extracted(self, parser):
        text = (
            "<tool_call>\n"
            "<function=a>x</function>\n"
            "<function=b>y</function>\n"
            "</tool_call>"
        )
        calls = parser._get_function_calls(text)
        assert len(calls) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
