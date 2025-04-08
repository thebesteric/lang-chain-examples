from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM, ChatOllama

# llm = OllamaLLM(model='qwen2.5:7b', temperature=0.0)
llm = ChatOllama(model='qwen2.5:7b', temperature=0.0)

messages = [
    SystemMessage(content="将以下内容翻译为中文"),
    HumanMessage(content="It's a nice day today"),
]

parse = StrOutputParser()

# 调用模型
result = llm.invoke(messages)
print(result)

response = parse.invoke(result)
print(response)