"""
GLM-4.7 Tool Call Parser.
Ported from vllm/tool_parsers/glm47_moe_tool_parser.py with vllm dependencies removed.

GLM-4.7 uses a slightly different tool call format compared to GLM-4:
  - The function name may appear on the same line as ``<tool_call>`` without
    a newline separator before the first ``<arg_key>``.
  - Tool calls may have zero arguments
    (e.g. ``<tool_call>func</tool_call>``).

This parser overrides the parent regex patterns to handle both formats.
"""

import logging
import re
from typing import Any

from .glm4_moe_tool_parser import Glm4MoeToolParser

logger = logging.getLogger(__name__)


class Glm47MoeToolParser(Glm4MoeToolParser):
    """Tool parser for GLM-4.7 models.

    Inherits all streaming and non-streaming logic from Glm4MoeToolParser,
    only overriding the regex patterns to support GLM-4.7's more flexible format.
    """

    supports_reasoning = False

    def __init__(self, tokenizer: Any, tools: list[dict] | None = None):
        super().__init__(tokenizer, tools)
        # GLM-4.7 format: <tool_call>func_name[<arg_key>...]*</tool_call>
        # The function name can be followed by a newline, whitespace, or
        # directly by <arg_key> tags (no separator).  The arg section is
        # optional so that zero-argument calls are supported.
        self.func_detail_regex = re.compile(
            r"<tool_call>\s*(\S+?)\s*(<arg_key>.*)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
