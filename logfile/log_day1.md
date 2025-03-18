# 📅 2025-03-18 项目日志

## ✅ **今日进展**
### **1. 实现 LangChain API 调用**
- 选用 `DeepSeek-Reasoner` 作为模型，进行医疗问答
- 通过 `langchain_openai.ChatOpenAI()` 进行 API 交互
- 实现 **流式输出**，提升用户体验

**📌 代码示例**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    model='deepseek-reasoner',
    temperature=0.3,
    api_key="your-api-key",
    streaming=True
)

response = llm.invoke("我头痛怎么办？")
print(response.content)
```

### **2.🧐 遇到的问题 & 解决方案**
**1 问题：API 断连导致崩溃**

原因：DeepSeek API 在请求过多时可能超时
解决方案：增加自动重试机制 (tenacity)

**2 问题：历史记录过长，导致 Token 超限**

解决方案：限制最多 5 轮对话，防止 Token 过载

**3 问题:流式对话 vs. 单次对话**

单次对话会等待模型生成完整回答后才显示，而流式对话可以边生成边显示，提高交互体验。

**单次对话**
```python
response = llm.invoke(question)
print(response.content)
```

**流式对话**
```python
print("\n回答：", end="", flush=True)
response = ""
for chunk in llm.stream(question):
    print(chunk.content, end="", flush=True)
    response += chunk.content
```
**4 问题:为什么对话顺序重要**

在 LangChain 里，我们使用 ChatPromptTemplate.from_messages() 组合系统消息（system）、用户消息（user）、AI 消息（assistant）。设计 messages 结构时，确保对话顺序正确。
```python
messages = [
    SystemMessage(content="你是一个专业的医学助手"),
    HumanMessage(content="我头痛怎么办？"),
    AIMessage(content="头痛可能有多种原因，如感冒、压力等...")
]
```

### **3.🎯 明日计划**
- **任务 3：优化用户交互体验**
  - 添加日志（`logging`），记录 API 失败详情
  - 提供更清晰的错误提示
  - 美化 CLI 交互体验（如 `print("⚠️ 服务器繁忙，请稍后再试！")`）

### **4.📚 学习心得 & 反思**
- `tenacity` 的 **指数退避重试**（exponential backoff）很重要，防止 API 过载
- 适当控制历史对话，减少 Token 负担
- LangChain 的 `ChatPromptTemplate.from_messages()` 需要**传递正确的格式**