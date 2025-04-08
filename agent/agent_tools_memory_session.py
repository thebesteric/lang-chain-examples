from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent.tools_tavily import tavily_search_tool
from agent.tools_wiki_search import wiki_retriever_tool

llm = ChatOpenAI(model="qwen2.5:7b", temperature=0.0, api_key=SecretStr("7RQ11EA-KJE42KB-M0EEJZZ-443YRDT"),
                 base_url="http://localhost:11434/v1")

# 定义工具集
tools = [
    # 维基百科查询工具
    wiki_retriever_tool,
    # Tavily 查询工具
    tavily_search_tool
]

store = {}

from langchain_community.chat_message_histories import ChatMessageHistory


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_tool_calling_agent(llm, tools, prompt)
# 创建代理执行器
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 使用 RunnableWithMessageHistory 携带 session
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_key="input",
    history_key="chat_history",
)

response = agent_with_chat_history.invoke(
    {"input": "我的名字是 Jack"},
    config={"configurable": {"session_id": "abc-123"}}
)
print(response)

response = agent_with_chat_history.invoke(
    {"input": "我叫什么名字"},
    config={"configurable": {"session_id": "abc-456"}}
)
print(response)


response = agent_with_chat_history.invoke(
    {"input": "我叫什么名字"},
    config={"configurable": {"session_id": "abc-123"}}
)
print(response)