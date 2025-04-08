from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model='qwen2.5:7b', temperature=0.0)
# 使用 load_tools 加载维基百科工具
tools = load_tools(["wikipedia"])
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手."),
        ("user", "{input}"),
        ("user", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    stream_runnable=False
)

with get_openai_callback() as cb:
    response = agent_executor.invoke(
        {"input": "土豆的学名是什么，哪种鸟飞的最快？"}
    )
    print(response)
    print("=" * 50)
    print(f"总令牌数: {cb.total_tokens}")
    print(f"提示令牌数: {cb.prompt_tokens}")
    print(f"完成令牌数: {cb.completion_tokens}")
    print(f"总花费: {cb.total_cost}")

"""
{'input': '土豆的学名是什么，哪种鸟飞的最快？', 'output': '土豆的学名是 Solanum tuberosum。\n\n最快的鸟是游隼，它的飞行速度可以达到每小时320公里以上。'}
==================================================
总令牌数: 1004
提示令牌数: 928
完成令牌数: 76
"""