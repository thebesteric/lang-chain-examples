from langchain.agents import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

db = SQLDatabase.from_uri("sqlite:///db/langchain.sqlite")
llm = ChatOpenAI(model="qwen2.5:7b", temperature=0.0, api_key=SecretStr("Ollama"), base_url="http://localhost:11434/v1")

db_tool_kit = SQLDatabaseToolkit(db=db, llm=llm)
print("tools", db_tool_kit.get_tools())

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=db_tool_kit,
    verbose=True,
    # AgentType.OPENAI_FUNCTIONS：该类型借鉴了 OpenAI 的函数调用机制，允许代理与外部工具或函数进行交互。代理能够根据输入的自然语言任务
    # AgentType.ZERO_SHOT_REACT_DESCRIPTION：零样本反应描述代理，不需要提供示例就能根据自然语言输入和工具的描述来决定何时使用工具以及如何使用
    # AgentType.ONE_SHOT_REACT_DESCRIPTION：单样本反应描述代理，与零样本代理类似，但需要提供一个示例来帮助代理更好地理解任务和工具的使用方式
    # AgentType.REACT_DOCSTORE：主要用于与文档存储和检索相关的任务，结合了反应（REACT）模式和文档存储的功能
    # AgentType.CONVERSATIONAL_REACT_DESCRIPTION：专门用于处理对话式任务的代理类型，能够理解和处理多轮对话中的自然语言输入，并根据对话的上下文和历史信息来决定工具的使用和回复内容
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

result = agent_executor.invoke({"input": "查询张三学了哪些课程"})
# result = agent_executor.invoke({"input": "查询姓名为张三用户信息"})
# result = agent_executor.invoke({"input": "描述一下 user 表的表结构信息"})
print("result: ", result)