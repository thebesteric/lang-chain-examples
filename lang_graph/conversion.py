from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List

from pydantic import SecretStr

memory = ConversationBufferMemory()

# 定义对话状态（包含历史记录和用户输入）
class AgentState(TypedDict):
    messages: List[dict]  # 对话历史
    user_input: str       # 当前用户输入

# 初始化图
workflow = StateGraph(AgentState)

llm = ChatOpenAI(model="qwen2.5:7b", temperature=0.0, api_key=SecretStr("ollama"), base_url="http://localhost:11434/v1")

# 定义节点函数
def process_input(state: AgentState):
    # 将用户输入添加到历史
    user_input = state["user_input"]
    memory.chat_memory.add_user_message(user_input)
    # 调用大模型生成回复（此处简化，实际需调用 OpenAI/Gemini）
    ai_response = llm.invoke(user_input)
    memory.chat_memory.add_ai_message(ai_response)
    return {"messages": memory.load_memory_variables({})["history"], "user_input": ""}

# 添加节点
workflow.add_node("process_input", process_input)
workflow.set_entry_point("process_input")
workflow.add_edge("process_input", END)


def call_agent(input):
    # 从记忆加载历史
    history = memory.load_memory_variables({})["history"]
    # 更新状态
    state = AgentState(messages=history, user_input=input)
    # 运行 LangGraph 流程
    for _ in workflow.compile().stream(state):
        pass
    # 获取最新回复
    response = memory.chat_memory.messages[-1].content
    print(response)


if __name__ == '__main__':
    call_agent("我叫jack")
    call_agent("我叫什么名字")

