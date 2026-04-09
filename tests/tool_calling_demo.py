import json
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    # base_url="http://10.146.225.70:18901/v1",
    base_url="http://127.0.0.1:18901/v1",
    # base_url="http://10.146.236.83:30000/v1",
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


def dump_response(response):
    """将完整的响应对象以格式化JSON打印出来"""
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))


def chat_with_tools(query):
    print(f"{'=' * 60}")
    print(f"=== 用户问题 ===\n{query}\n")

    messages = [{"role": "user", "content": query}]

    # 第一轮：发送带工具定义的请求
    print(">>> 第一轮请求 messages >>>")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    print()

    response = client.chat.completions.create(
        model="claude-debug",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=1.0,
        max_tokens=131072,
    )

    print("<<< 第一轮完整响应 JSON <<<")
    dump_response(response)
    print()

    assistant_message = response.choices[0].message

    # 如果没有工具调用，直接返回
    if not assistant_message.tool_calls:
        print(f"模型未调用工具，直接回答：\n{assistant_message.content}")
        print("=" * 60 + "\n")
        return

    # 将 assistant 消息加入对话历史
    messages.append(assistant_message.model_dump())

    # 第二轮：执行工具并将结果返回给模型
    for tool_call in assistant_message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        print(f"--- 执行工具: {func_name}({json.dumps(func_args, ensure_ascii=False)}) ---")

        result = execute_tool(func_name, func_args)
        print(f"--- 工具返回: {result} ---\n")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # 打印第二轮完整的请求 messages
    print(">>> 第二轮请求 messages >>>")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    print()

    # 第三轮：模型根据工具结果生成最终回答
    final_response = client.chat.completions.create(
        model="qwen35",
        messages=messages,
        tools=tools,
        temperature=1.0,
        max_tokens=131072,
    )

    print("<<< 第二轮完整响应 JSON <<<")
    dump_response(final_response)
    print()

    print(f"=== 最终回答 ===\n{final_response.choices[0].message.content}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 测试1：单工具调用
    chat_with_tools("今天北京天气怎么样？")
    exit()

    # 测试2：多工具调用（parallel tool calls）
    chat_with_tools("我想知道上海的天气，另外帮我查一下明天从北京到上海的航班")

    # 测试3：不需要工具的普通问题
    chat_with_tools("你好，请介绍一下你自己")
