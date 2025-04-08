import asyncio

from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model='qwen2.5:7b', temperature=0.0)
prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
parse = StrOutputParser()

chain = prompt | llm | parse


# 定义异步函数
async def async_stream(topic):
    # 异步流式输出结果
    async for chunk in chain.astream({"topic": topic}):
        print(chunk, end="|", flush=True)

# 运行异步函数
# asyncio.run(async_stream("猫"))

# 定义主异步函数
async def main():
    # 创建多个异步任务
    tasks = [async_stream("猫"), async_stream("狗")]
    # tasks = [async_stream("猫")]
    # 并发运行所有任务
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())