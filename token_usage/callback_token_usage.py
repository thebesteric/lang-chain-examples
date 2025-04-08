from langchain_community.callbacks import get_openai_callback
from langchain_ollama import ChatOllama

llm = ChatOllama(model='qwen2.5:7b', temperature=0.0)
with get_openai_callback() as cb:
    result = llm.invoke("你好")
    print(result)
    print("=" * 50)
    print(cb)

"""
Tokens Used: 41
	Prompt Tokens: 30
		Prompt Tokens Cached: 0
	Completion Tokens: 11
		Reasoning Tokens: 0
Successful Requests: 1
Total Cost (USD): $0.0
"""
