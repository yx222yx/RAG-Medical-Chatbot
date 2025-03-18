import os
import time
from dotenv import load_dotenv
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 限制最大历史轮数
MAX_HISTORY = 5

def load_environment():
    env_path = Path(__file__).parent / "DEEPSEEKAPI.env"

    if not env_path.exists():
        raise FileNotFoundError(f"环境文件 {env_path} 不存在")

    load_dotenv(env_path)
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("环境文件中未找到 DEEPSEEK_API_KEY")

    return api_key

def initialize_model():
    return ChatOpenAI(
        base_url="https://api.deepseek.com/v1",
        model='deepseek-reasoner',
        temperature=0.3,
        api_key=load_environment(),
        max_retries=3,
        streaming=True
    )

# 验证消息顺序
def validate_messages(messages):
    last_role = None
    for msg in messages:
        current_role = msg[0]
        if current_role == "system":
            continue
        if current_role == last_role:
            raise ValueError(f"连续的同类型消息：{last_role} -> {current_role}")
        last_role = current_role

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_api_with_retry(llm, messages):

    formatted_messages = ChatPromptTemplate.from_messages(messages).format_messages()
    return llm.stream(formatted_messages)

def main():
    print(">>> 正在初始化问答助手...")
    history = []
    llm = initialize_model()

    try:
        print(">>> 助手已就绪（输入 q 退出）")
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        return

    while True:
        try:
            question = input("\n提问：").strip()
            if question.lower() in ['q', 'quit', 'exit']:
                break
            if not question:
                continue


            messages = [
                SystemMessage(content="你是一个专业的中文助手，回答需准确简洁")
            ] + [
                HumanMessage(content=msg[1]) if msg[0] == "user" else AIMessage(content=msg[1]) for msg in history
            ] + [
                HumanMessage(content=question)
            ]

            if len(history) > MAX_HISTORY * 2:
                history = history[-MAX_HISTORY * 2:]

            try:
                validate_messages([(msg.type, msg.content) for msg in messages])
            except ValueError as e:
                print(f"输入错误：{str(e)}")
                continue

            # 流式输出
            print("\n回答：", end="", flush=True)
            response = ""
            for chunk in call_api_with_retry(llm, messages):
                print(chunk.content, end="", flush=True)
                response += chunk.content

            history.extend([
                ("user", question),
                ("assistant", response)
            ])
            print()

        except KeyboardInterrupt:
            print("\n对话已终止")
            break
        except Exception as e:
            print(f"请求出错：{str(e)}")


if __name__ == "__main__":
    main()
