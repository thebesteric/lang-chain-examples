import getpass
import os

from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.globals import set_verbose, set_debug
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 导入查询工具
from tools_tavily import tavily_search_tool
# 导入维基百科查询工具
from tools_wiki_search import wiki_retriever_tool

if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("LANGSMITH API key:\n")

# 定义模型
llm = ChatOpenAI(model="qwen2.5:7b", temperature=0.0, api_key=SecretStr("ollama"), base_url="http://localhost:11434/v1")


# 定义工具集
tools = [
    # Tavily 查询工具
    tavily_search_tool,
    # 维基百科查询工具
    wiki_retriever_tool,
]

# https://smith.langchain.com/hub/hwchase17/openai-functions-agent
# SYSTEM
# You are a helpful assistant
# PLACEHOLDER
# chat_history
# HUMAN
# {input}
# PLACEHOLDER
# agent_scratchpad
prompt = hub.pull("hwchase17/openai-functions-agent")
# [
# SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='You are a helpful assistant'), additional_kwargs={}),
# MessagesPlaceholder(variable_name='chat_history', optional=True),
# HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], input_types={}, partial_variables={}, template='{input}'), additional_kwargs={}),
# MessagesPlaceholder(variable_name='agent_scratchpad')
# ]
print(prompt.messages)

# 创建代理
agent = create_tool_calling_agent(llm, tools, prompt)
# 创建代理执行器
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 打印详细日志
set_verbose(True)

# 打印调试日志（更加详细）
# set_debug(True)

# 执行
# print(agent_executor.invoke({"input": "猫的特征？"}))
# print(agent_executor.invoke({"input": "今天合肥天气如何？"}))
print(agent_executor.invoke({
    # "chat_history": [
    #     HumanMessage(content="我的名字是 Jack"),
    #     AIMessage(content="你好，Jack，很高兴认识你")
    # ],
    "input": "猫的特征？今天合肥天气如何？",
}))

# {'chat_history': [HumanMessage(content='我的名字是 Jack', additional_kwargs={}, response_metadata={}),
# AIMessage(content='你好，Jack，很高兴认识你', additional_kwargs={}, response_metadata={})],
# 'input': '我的名字是什么？',
# 'output': '你的名字是Jack。'}
