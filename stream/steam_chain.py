from langchain_ollama import ChatOllama

llm = ChatOllama(model='qwen2.5:7b', temperature=0.0)

chunks = []
for chunk in llm.stream("天空是什么颜色的？"):
    chunks.append(chunk)
    print(chunk, end="|", flush=True)