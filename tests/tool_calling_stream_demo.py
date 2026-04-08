"""
流式(stream)模式的多轮 agent 测试样例。
仿照 tool_calling_demo.py，使用 stream=True 进行工具调用的多轮对话。
"""

import json
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:18901/v1",
)

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "搜索两个城市之间的航班信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {
                        "type": "string",
                        "description": "出发城市",
                    },
                    "destination": {
                        "type": "string",
                        "description": "目的城市",
                    },
                    "date": {
                        "type": "string",
                        "description": "出发日期，格式为 YYYY-MM-DD",
                    },
                },
                "required": ["departure", "destination", "date"],
            },
        },
    },
]


# 模拟工具执行结果
def execute_tool(name, arguments):
    if name == "get_weather":
        city = arguments.get("city", "未知")
        return json.dumps({
            "city": city,
            "temperature": 22,
            "unit": "celsius",
            "condition": "晴天",
            "humidity": "45%",
        }, ensure_ascii=False)
    elif name == "search_flights":
        return json.dumps({
            "flights": [
                {"flight": "CA1234", "departure_time": "08:00", "arrival_time": "11:00", "price": 1200},
                {"flight": "MU5678", "departure_time": "14:30", "arrival_time": "17:30", "price": 980},
            ]
        }, ensure_ascii=False)
    return json.dumps({"error": "未知工具"})


def collect_stream_response(stream):
    """
    从流式响应中收集完整的 assistant 消息。
    返回: (content, tool_calls, raw_chunks)
      - content: 拼接后的文本内容
      - tool_calls: dict[index -> {id, name, arguments}] 累积的工具调用
      - raw_chunks: 所有原始 chunk 的 dump 列表
    """
    content = ""
    tool_calls = {}  # index -> {id, type, function: {name, arguments}}
    raw_chunks = []
    finish_reason = None

    for chunk in stream:
        raw_chunks.append(chunk.model_dump())
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        # 累积文本内容
        if delta.content:
            content += delta.content
            print(delta.content, end="", flush=True)

        # 累积工具调用（流式 tool_calls 是增量的）
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc_delta.id:
                    tool_calls[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls[idx]["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    print()  # 换行
    return content, tool_calls, raw_chunks, finish_reason


def stream_chat_with_tools(query):
    """流式模式的多轮工具调用 agent 循环"""
    print(f"{'=' * 60}")
    print(f"=== 用户问题 ===\n{query}\n")

    messages = [{"role": "user", "content": query}]
    round_num = 0
    max_rounds = 5  # 防止无限循环

    while round_num < max_rounds:
        round_num += 1
        print(f">>> 第 {round_num} 轮请求 messages >>>")
        print(json.dumps(messages, indent=2, ensure_ascii=False, default=str))
        print()

        # 发送流式请求
        print(f"<<< 第 {round_num} 轮流式响应 <<<")
        stream = client.chat.completions.create(
            model="claude-debug",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=1.0,
            max_tokens=131072,
            stream=True,
        )

        content, tool_calls, raw_chunks, finish_reason = collect_stream_response(stream)

        # 打印收集到的完整信息
        print(f"\n[finish_reason={finish_reason}]")
        if tool_calls:
            print(f"[收集到 {len(tool_calls)} 个工具调用]")
            for idx in sorted(tool_calls.keys()):
                tc = tool_calls[idx]
                print(f"  [{idx}] {tc['function']['name']}({tc['function']['arguments']})")
        print()

        # 打印所有原始 chunks（可选，用于调试）
        # for i, c in enumerate(raw_chunks):
        #     print(f"  chunk[{i}]: {json.dumps(c, ensure_ascii=False)}")

        # 构造 assistant 消息加入历史
        assistant_msg = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant_msg["tool_calls"] = [tool_calls[idx] for idx in sorted(tool_calls.keys())]
        messages.append(assistant_msg)

        # 如果没有工具调用，对话结束
        if not tool_calls:
            print(f"=== 最终回答 ===\n{content}")
            print("=" * 60 + "\n")
            return

        # 执行工具并将结果加入消息历史
        for idx in sorted(tool_calls.keys()):
            tc = tool_calls[idx]
            func_name = tc["function"]["name"]
            func_args = json.loads(tc["function"]["arguments"])
            print(f"--- 执行工具: {func_name}({json.dumps(func_args, ensure_ascii=False)}) ---")

            result = execute_tool(func_name, func_args)
            print(f"--- 工具返回: {result} ---\n")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    print("[WARNING] 达到最大轮数限制，停止循环")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 测试1：单工具调用（流式）
    stream_chat_with_tools("今天北京天气怎么样？")

    # 测试2：多工具调用 parallel tool calls（流式）
    stream_chat_with_tools("我想知道上海的天气，另外帮我查一下明天从北京到上海的航班")

    # 测试3：不需要工具的普通问题（流式）
    stream_chat_with_tools("你好，请介绍一下你自己")
