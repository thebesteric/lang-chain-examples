import asyncio

from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# llm = Ollama(model='deepseek-r1:8b', temperature=0.0)
llm = ChatOllama(model='qwen2.5:7b', temperature=0.0)
parse = JsonOutputParser()

chain = llm | parse


# 定义异步函数
async def async_stream():
    # 异步流式输出结果
    async for chunk in chain.astream(
            """
            以JSON格式输出法国、西班牙和日本的国家及其人口列表。
            使用一个带有 "countries" 键的对象，其中包含国家列表。
            每个国家都应该有"name"和"population"键。
            """):
        print(chunk, end="\n", flush=True)


# 运行异步函数
asyncio.run(async_stream())

# {'countries': [{'name': 'France', 'population': '67200000'}, {'name': 'Spain', 'population': '47300000'}, {'name': 'Japan', 'population': '125100000'}]}
