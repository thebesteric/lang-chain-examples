import asyncio

from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

# llm = Ollama(model='deepseek-r1:8b', temperature=0.0)
llm = ChatOllama(model='qwen2.5:7b', temperature=0.0)

parse = JsonOutputParser()

chain = llm | parse

async def async_stream():
    async for event in chain.astream_events(input="hello", version="v2"):
        print(event)

"""
on_chain_start
on_chain_end
on_chat_model_start
on_chat_model_end
on_chat_model_stream
on_chat_model_invoke
on_parser_start
on_parser_end
on_tool_start
on_tool_end
on_retriever_strat
on_retriever_end
on_prompt_start
on_prompt_end
"""
asyncio.run(async_stream())
