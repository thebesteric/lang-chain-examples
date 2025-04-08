import os

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.globals import set_verbose, set_debug
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

os.environ['LANGSMITH_PROJECT'] = 'langserve_app'

llm = ChatOpenAI(model="qwen2.5:7b", temperature=0.0, api_key=SecretStr("EMPTY"), base_url="http://localhost:11434/v1")
tools = [TavilySearchResults(max_results=1)]
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位得力的助手"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# 构建工具代理
agent = create_tool_calling_agent(llm, tools, prompt)
# 打印详细日志
set_debug(True)

# 打印关键日志
# set_verbose(True)

# 通过传入 agent 和 tools 创建代理执行器
agent_executor = AgentExecutor(agent=agent, tools=tools)
response = agent_executor.invoke(
    {"input": "谁执导了2024年的电影《哪吒之魔童闹海》,他多少岁了？"}
)
print(response)