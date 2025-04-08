from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


@tool
def tone_tool(tone: Literal["温柔的", "严厉的", "普通的"]):
    """
    Describe the weather
    """
    print(f"===> call tools: tone_tool, tone: {tone}")
    pass


llm = ChatOpenAI(model='qwen2.5:7b', base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))
llm_with_tools = llm.bind_tools([tone_tool])

human_message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "描述一下这两句话的语气",
        },
        {
            "type": "text",
            "text": "我可以请你看电影么",
        },
        {
            "type": "text",
            "text": "做不完作业，就不许吃饭",
        }
    ]
)

response = llm_with_tools.invoke([human_message])
print(response.tool_calls)

"""
[
    {'name': 'tone_tool', 'args': {'tone': '温柔的'}, 'id': 'call_rr735hng', 'type': 'tool_call'}, 
    {'name': 'tone_tool', 'args': {'tone': '严厉的'}, 'id': 'call_7vdt2fq6', 'type': 'tool_call'}
]
"""
