from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent.tools_tavily import tavily_search_tool
from agent.tools_wiki_search import wiki_retriever_tool

llm = ChatOpenAI(model="qwen2.5:7b", temperature=0.0, api_key=SecretStr("SECRET"), base_url="http://localhost:11434/v1")

# 定义工具集
tools = [
    # 维基百科查询工具
    wiki_retriever_tool,
    # Tavily 查询工具
    tavily_search_tool
]

prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_tool_calling_agent(llm, tools, prompt)
# 创建代理执行器
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 执行
print(agent_executor.invoke({
    "chat_history": [
        HumanMessage(content="我的名字是 Jack"),
        AIMessage(content="你好，Jack，很高兴认识你")
    ],
    "input": "我的名字是什么？",
}))
