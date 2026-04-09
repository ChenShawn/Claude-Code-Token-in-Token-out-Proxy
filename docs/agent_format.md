## Agent TrajectoryStore 数据存储格式

```
Claude-Code → LiteLLM  →  [Proxy: FastAPI]  →  SGLang
                                 ↓
                          Tokenizer (本地)
                                 ↓
                          TrajectoryDumper
```

TrajectoryStore在 global_types.py 文件中定义，层级关系为：
1. 一个TrajectoryStore对象对应一条trajectory，一个trajectory有一个唯一的traj_id来标识；
2. 一条trajectory可以包含多个agent，每个agent是一条从头到尾完整的多轮交互轨迹，每个agent有一个唯一的agent_id标识；
3. 一条trajectory中的多个agent之间存在顺序关系；
4. 当client通过litellm给proxy发消息时：
(a) 如果这条消息的前缀和当前任意一条agent的前缀完全匹配，视为对应那一个agent的多轮交互请求，sglang的返回结果应当append到agent历史后面；
(b) 如果目前没有任何一条agent trajectory能够前缀匹配到已有的agent，那么创建一个新的agent存储到trajectoryStore中；

TrajectoryStore可以导出成JSONL或parquet两种文件格式，分别对应两个不同的函数接口，不要实现成同一个函数：
1. JSONL：只存储OpenAI格式的chat messages，每行对应一个agent，必须按顺序存储，同时记录存储agent_id，不存储token id；
2. parquet：同时存储OpenAI格式的chat messages和token ids，每行对应一个agent，必须按顺序存储，同时记录存储agent_id；
