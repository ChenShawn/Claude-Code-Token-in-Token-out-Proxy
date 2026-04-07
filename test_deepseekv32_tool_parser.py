"""
Tests for DeepSeekV32ToolParser.
Uses a mock tokenizer to avoid HuggingFace model downloads.

DSML format uses fullwidth pipes (U+FF5C):
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="func">
    <｜DSML｜parameter name="p" string="true">val</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
"""

import json
import pytest

from deepseekv32_tool_parser import DeepSeekV32ToolParser
from tool_types import DeltaMessage

# Fullwidth pipe shorthand
FP = "\uff5c"

# DSML tag helpers
TAG_FC_OPEN = f"<{FP}DSML{FP}function_calls>"
TAG_FC_CLOSE = f"</{FP}DSML{FP}function_calls>"
TAG_INVOKE_OPEN = f"<{FP}DSML{FP}invoke"  # needs name="..." > appended
TAG_INVOKE_CLOSE = f"</{FP}DSML{FP}invoke>"
TAG_PARAM_OPEN = f"<{FP}DSML{FP}parameter"  # needs name="..." string="..."> appended
TAG_PARAM_CLOSE = f"</{FP}DSML{FP}parameter>"


def _invoke(name: str, params: list[tuple[str, str, str]]) -> str:
    """Build an invoke block. params is list of (name, string_flag, value)."""
    lines = [f'{TAG_INVOKE_OPEN} name="{name}">']
    for pname, sflag, pval in params:
        lines.append(
            f'{TAG_PARAM_OPEN} name="{pname}" string="{sflag}">{pval}{TAG_PARAM_CLOSE}'
        )
    lines.append(TAG_INVOKE_CLOSE)
    return "\n".join(lines)


def _function_calls(*invocations: str) -> str:
    """Wrap invocations in function_calls tags."""
    body = "\n".join(invocations)
    return f"{TAG_FC_OPEN}\n{body}\n{TAG_FC_CLOSE}"


# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    def __init__(self):
        self._vocab = {
            f"<{FP}DSML{FP}function_calls>": 200,
            f"</{FP}DSML{FP}function_calls>": 201,
            "hello": 1,
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
                "temperature": {"type": "number"},
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
    return DeepSeekV32ToolParser(MockTokenizer(), tools=ALL_TOOLS)


# ===================================================================
# Non-streaming: extract_tool_calls
# ===================================================================

class TestExtractToolCalls:

    def test_no_tool_call(self, parser):
        result = parser.extract_tool_calls("Hello, how can I help?", None)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Hello, how can I help?"

    def test_single_tool_call_one_param(self, parser):
        output = _function_calls(
            _invoke("get_weather", [("location", "true", "San Francisco")])
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert args["location"] == "San Francisco"

    def test_single_tool_call_multiple_params(self, parser):
        output = _function_calls(
            _invoke("get_weather", [
                ("location", "true", "Tokyo"),
                ("unit", "true", "celsius"),
            ])
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "Tokyo"
        assert args["unit"] == "celsius"

    def test_content_before_tool_call(self, parser):
        prefix = "Let me check the weather.\n"
        output = prefix + _function_calls(
            _invoke("get_weather", [("location", "true", "Paris")])
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert result.content == prefix

    def test_content_is_none_when_tool_at_start(self, parser):
        output = _function_calls(
            _invoke("get_weather", [("location", "true", "Berlin")])
        )
        result = parser.extract_tool_calls(output, None)
        assert result.content is None

    def test_multiple_invocations_in_one_block(self, parser):
        output = _function_calls(
            _invoke("get_weather", [("location", "true", "London")]),
            _invoke("calculate", [("expression", "true", "2+2")]),
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "calculate"

    def test_multiple_function_calls_blocks(self, parser):
        output = (
            _function_calls(
                _invoke("get_weather", [("location", "true", "NYC")])
            )
            + "\n"
            + _function_calls(
                _invoke("calculate", [("expression", "true", "3*3")])
            )
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2

    def test_no_parameters(self, parser):
        output = _function_calls(
            f'{TAG_INVOKE_OPEN} name="get_weather">\n{TAG_INVOKE_CLOSE}'
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

    def test_unknown_function(self, parser):
        output = _function_calls(
            _invoke("unknown_func", [("foo", "true", "bar")])
        )
        result = parser.extract_tool_calls(output, None)
        assert result.tools_called is True
        tc = result.tool_calls[0]
        assert tc.function.name == "unknown_func"
        args = json.loads(tc.function.arguments)
        assert args["foo"] == "bar"

    def test_tool_call_id_format(self, parser):
        output = _function_calls(
            _invoke("get_weather", [("location", "true", "LA")])
        )
        result = parser.extract_tool_calls(output, None)
        tc_id = result.tool_calls[0].id
        assert tc_id.startswith("call_")
        hex_part = tc_id[5:]
        assert len(hex_part) == 24
        int(hex_part, 16)  # should not raise


# ===================================================================
# Parameter type conversion
# ===================================================================

class TestParamConversion:

    def test_string_param(self, parser):
        output = _function_calls(
            _invoke("get_weather", [("location", "true", "Boston")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "Boston"
        assert isinstance(args["location"], str)

    def test_integer_param(self, parser):
        output = _function_calls(
            _invoke("calculate", [("precision", "false", "5")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["precision"] == 5
        assert isinstance(args["precision"], int)

    def test_boolean_true(self, parser):
        output = _function_calls(
            _invoke("calculate", [("verbose", "false", "true")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["verbose"] is True

    def test_boolean_false(self, parser):
        output = _function_calls(
            _invoke("calculate", [("verbose", "false", "false")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["verbose"] is False

    def test_boolean_from_0_and_1(self, parser):
        output1 = _function_calls(
            _invoke("calculate", [("verbose", "false", "1")])
        )
        result1 = parser.extract_tool_calls(output1, None)
        args1 = json.loads(result1.tool_calls[0].function.arguments)
        assert args1["verbose"] is True

        p2 = DeepSeekV32ToolParser(MockTokenizer(), tools=ALL_TOOLS)
        output0 = _function_calls(
            _invoke("calculate", [("verbose", "false", "0")])
        )
        result0 = p2.extract_tool_calls(output0, None)
        args0 = json.loads(result0.tool_calls[0].function.arguments)
        assert args0["verbose"] is False

    def test_number_float(self, parser):
        output = _function_calls(
            _invoke("calculate", [("temperature", "false", "3.14")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["temperature"] == 3.14

    def test_number_integer_valued_float(self, parser):
        """5.0 should become int 5 for 'number' type."""
        output = _function_calls(
            _invoke("calculate", [("temperature", "false", "5.0")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["temperature"] == 5
        assert isinstance(args["temperature"], int)

    def test_null_value(self, parser):
        output = _function_calls(
            _invoke("get_weather", [("location", "true", "null")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] is None

    def test_array_param(self, parser):
        output = _function_calls(
            _invoke("search", [("scores", "false", "[0.9, 0.8, 0.7]")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["scores"] == [0.9, 0.8, 0.7]

    def test_object_param(self, parser):
        output = _function_calls(
            _invoke("search", [("filters", "false", '{"category": "news"}')])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["filters"] == {"category": "news"}

    def test_invalid_integer_falls_back_to_string(self, parser):
        """If integer conversion fails, should fall back to string."""
        output = _function_calls(
            _invoke("calculate", [("precision", "false", "not_a_number")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["precision"] == "not_a_number"

    def test_invalid_boolean_falls_back_to_string(self, parser):
        """Invalid boolean falls back to string."""
        output = _function_calls(
            _invoke("calculate", [("verbose", "false", "maybe")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["verbose"] == "maybe"

    def test_unknown_param_defaults_to_string(self, parser):
        """Param not in schema defaults to string type."""
        output = _function_calls(
            _invoke("get_weather", [("unknown_param", "true", "some_value")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["unknown_param"] == "some_value"


# ===================================================================
# _convert_param_value / _convert_param_value_checked
# ===================================================================

class TestConvertParamValue:

    def test_null_returns_none(self, parser):
        assert parser._convert_param_value("null", "string") is None
        assert parser._convert_param_value("NULL", "integer") is None
        assert parser._convert_param_value("Null", "boolean") is None

    def test_string_types(self, parser):
        for t in ["string", "str", "text"]:
            assert parser._convert_param_value("hello", t) == "hello"

    def test_integer_types(self, parser):
        assert parser._convert_param_value("42", "integer") == 42
        assert parser._convert_param_value("42", "int") == 42

    def test_number_types(self, parser):
        assert parser._convert_param_value("3.14", "number") == 3.14
        assert parser._convert_param_value("3.14", "float") == 3.14

    def test_boolean_types(self, parser):
        assert parser._convert_param_value("true", "boolean") is True
        assert parser._convert_param_value("false", "bool") is False

    def test_object_type(self, parser):
        assert parser._convert_param_value('{"a": 1}', "object") == {"a": 1}

    def test_array_type(self, parser):
        assert parser._convert_param_value("[1, 2]", "array") == [1, 2]

    def test_list_of_types_tries_in_order(self, parser):
        """When param_type is a list, tries each until one succeeds."""
        result = parser._convert_param_value("42", ["integer", "string"])
        assert result == 42

        result = parser._convert_param_value("hello", ["integer", "string"])
        assert result == "hello"

    def test_fallback_to_raw_string(self, parser):
        """If no type matches, returns raw string."""
        result = parser._convert_param_value("weird{data", ["object"])
        assert result == "weird{data"


# ===================================================================
# _convert_params_with_schema
# ===================================================================

class TestConvertParamsWithSchema:

    def test_with_matching_tool(self, parser):
        params = {"location": "NYC", "unit": "fahrenheit"}
        converted = parser._convert_params_with_schema("get_weather", params)
        assert converted["location"] == "NYC"
        assert converted["unit"] == "fahrenheit"

    def test_with_integer_type(self, parser):
        params = {"precision": "10"}
        converted = parser._convert_params_with_schema("calculate", params)
        assert converted["precision"] == 10

    def test_unknown_function_defaults_to_string(self, parser):
        params = {"foo": "bar"}
        converted = parser._convert_params_with_schema("nonexistent", params)
        assert converted["foo"] == "bar"

    def test_no_tools(self):
        p = DeepSeekV32ToolParser(MockTokenizer(), tools=None)
        params = {"x": "42"}
        converted = p._convert_params_with_schema("anything", params)
        assert converted["x"] == "42"


# ===================================================================
# _parse_invoke_params
# ===================================================================

class TestParseInvokeParams:

    def test_basic_params(self, parser):
        body = (
            f'{TAG_PARAM_OPEN} name="city" string="true">London{TAG_PARAM_CLOSE}\n'
            f'{TAG_PARAM_OPEN} name="count" string="false">5{TAG_PARAM_CLOSE}'
        )
        result = parser._parse_invoke_params(body)
        assert result == {"city": "London", "count": "5"}

    def test_no_params(self, parser):
        result = parser._parse_invoke_params("")
        assert result == {}

    def test_param_with_special_chars(self, parser):
        body = f'{TAG_PARAM_OPEN} name="q" string="true">what is "AI"?{TAG_PARAM_CLOSE}'
        result = parser._parse_invoke_params(body)
        assert result["q"] == 'what is "AI"?'


# ===================================================================
# Constructor
# ===================================================================

class TestConstructor:

    def test_requires_tokenizer(self):
        with pytest.raises(ValueError, match="model tokenizer must be passed"):
            DeepSeekV32ToolParser(None, tools=[])

    def test_requires_no_skip_special_tokens_flag(self):
        assert DeepSeekV32ToolParser.requires_no_skip_special_tokens is True

    def test_tools_default_to_empty(self):
        p = DeepSeekV32ToolParser(MockTokenizer())
        assert p.tools == []


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_param_value_with_newlines(self, parser):
        """Multiline parameter value."""
        body = f'{TAG_PARAM_OPEN} name="code" string="true">line1\nline2\nline3{TAG_PARAM_CLOSE}'
        invoke = f'{TAG_INVOKE_OPEN} name="search">\n{body}\n{TAG_INVOKE_CLOSE}'
        output = f"{TAG_FC_OPEN}\n{invoke}\n{TAG_FC_CLOSE}"
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["code"] == "line1\nline2\nline3"

    def test_incomplete_function_calls_block(self, parser):
        """Missing closing tag means regex won't match — no tool calls extracted."""
        output = (
            f"{TAG_FC_OPEN}\n"
            + _invoke("get_weather", [("location", "true", "Seattle")])
            # no TAG_FC_CLOSE
        )
        result = parser.extract_tool_calls(output, None)
        # The start token IS present, but the complete regex won't match
        assert result.tools_called is False

    def test_unicode_in_param_value(self, parser):
        output = _function_calls(
            _invoke("search", [("query", "true", "天気予報 東京")])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["query"] == "天気予報 東京"

    def test_mixed_params_types(self, parser):
        """Multiple params with different types in one call."""
        output = _function_calls(
            _invoke("calculate", [
                ("expression", "true", "2+2"),
                ("precision", "false", "3"),
                ("verbose", "false", "true"),
                ("temperature", "false", "0.7"),
            ])
        )
        result = parser.extract_tool_calls(output, None)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["expression"] == "2+2"
        assert args["precision"] == 3
        assert args["verbose"] is True
        assert args["temperature"] == 0.7


# ===================================================================
# Streaming: extract_tool_calls_streaming
# ===================================================================

class TestExtractToolCallsStreaming:

    def _simulate_stream_by_chunks(self, parser, chunks: list[str]):
        """Feed text in specified chunks, collecting non-None DeltaMessages."""
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
                request=None,
            )
            if msg is not None:
                results.append(msg)
        return results

    def test_plain_text_streaming(self, parser):
        results = self._simulate_stream_by_chunks(parser, ["Hello", " world"])
        content_parts = [r.content for r in results if r.content]
        assert "".join(content_parts) == "Hello world"
        assert all(r.tool_calls is None or r.tool_calls == [] for r in results)

    def test_tool_call_streaming(self, parser):
        invoke_block = _invoke("get_weather", [("location", "true", "Paris")])
        chunks = [
            "Let me check. ",
            TAG_FC_OPEN,
            "\n",
            invoke_block,
            "\n",
            TAG_FC_CLOSE,
        ]
        results = self._simulate_stream_by_chunks(parser, chunks)

        # Should have content before tool call
        content_msgs = [r for r in results if r.content and r.content.strip()]
        assert any("Let me check" in (m.content or "") for m in content_msgs)

        # Should have tool call messages
        tool_msgs = [r for r in results if r.tool_calls]
        assert len(tool_msgs) > 0

        # Verify tool call details
        all_tool_calls = []
        for r in tool_msgs:
            all_tool_calls.extend(r.tool_calls)
        assert any(tc.function.name == "get_weather" for tc in all_tool_calls)

        # Verify arguments
        for tc in all_tool_calls:
            if tc.function.name == "get_weather":
                args = json.loads(tc.function.arguments)
                assert args["location"] == "Paris"

    def test_streaming_resets_on_empty_previous(self, parser):
        """State resets when previous_text is empty."""
        # Simulate a partial stream
        parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="H",
            delta_text="H",
            previous_token_ids=[],
            current_token_ids=[1],
            delta_token_ids=[1],
            request=None,
        )
        assert parser.current_tool_index == 0

        # New stream resets
        parser.current_tool_index = 5  # artificially set
        parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="X",
            delta_text="X",
            previous_token_ids=[],
            current_token_ids=[2],
            delta_token_ids=[2],
            request=None,
        )
        assert parser.current_tool_index == 0

    def test_streaming_empty_delta_no_tool(self, parser):
        """Empty delta with no prior tool calls returns None."""
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

    def test_streaming_multiple_invocations(self, parser):
        """Stream with multiple invocations in one function_calls block."""
        invoke1 = _invoke("get_weather", [("location", "true", "NYC")])
        invoke2 = _invoke("calculate", [("expression", "true", "1+1")])
        full = _function_calls(invoke1, invoke2)

        # Feed in a few big chunks
        chunks = [full[:40], full[40:100], full[100:]]
        results = self._simulate_stream_by_chunks(parser, chunks)

        tool_msgs = [r for r in results if r.tool_calls]
        all_tool_calls = []
        for r in tool_msgs:
            all_tool_calls.extend(r.tool_calls)

        names = [tc.function.name for tc in all_tool_calls if tc.function.name]
        assert "get_weather" in names
        assert "calculate" in names

    def test_streaming_tool_call_ids_unique(self, parser):
        """Each streamed tool call should get a unique ID."""
        invoke1 = _invoke("get_weather", [("location", "true", "A")])
        invoke2 = _invoke("search", [("query", "true", "B")])
        full = _function_calls(invoke1, invoke2)

        chunks = [full[:50], full[50:]]
        results = self._simulate_stream_by_chunks(parser, chunks)

        all_tool_calls = []
        for r in results:
            if r.tool_calls:
                all_tool_calls.extend(r.tool_calls)

        ids = [tc.id for tc in all_tool_calls if tc.id]
        assert len(ids) == len(set(ids)), "Tool call IDs should be unique"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
