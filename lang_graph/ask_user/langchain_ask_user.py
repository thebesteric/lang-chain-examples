from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
import re

from langchain_openai import ChatOpenAI
from pydantic import SecretStr


# 自定义工具，用于向用户询问信息
class AskUserTool:
    def __init__(self, name="AskUser", description="Ask the user for specific information"):
        self.name = name
        self.description = description

    def run(self, question):
        # 这里可以实现与用户交互的逻辑，例如通过输入框获取用户输入
        user_input = input(question)
        return user_input


# 定义一个简单的工具列表
tools = [
    Tool(
        name="AskUser",
        func=AskUserTool().run,
        description="Ask the user for specific information"
    )
]

# 初始化聊天模型
llm = ChatOpenAI(model="qwen2.5:7b", temperature=0.0, api_key=SecretStr("ollama"), base_url="http://localhost:11434/v1")

# 定义 Agent 的提示模板
template = """You are a helpful assistant. If you need more information, you can use the 'AskUser' tool.

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=template
)

# 初始化 Agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, prompt=prompt)


# 示例问题，Agent 需要向用户询问信息
def run_agent():
    question = "Please tell me your name and age. First, tell me your name."
    result = agent.invoke({"input": question})
    print("result: ", result)
    name = result
    # 继续询问年龄
    age_question = f"Your name is {name}. Now, please tell me your age."
    age_result = agent.invoke({"input": age_question})
    print(f"Your name is {name} and your age is {age_result}.")


if __name__ == "__main__":
    run_agent()
