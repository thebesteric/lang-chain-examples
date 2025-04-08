from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def weather_tool(weather: Literal["晴朗的", "多云的", "多雨的", "下雪的"]):
    """Describe the weather"""
    pass


# 需要模型有视觉支持，如 deepseek-ai/deepseek-vl2，Qwen/Qwen2.5-VL-7B-Instruct
model = ChatOpenAI(model='gpt-4o', temperature=0.0)

model_with_tools = model.bind_tools([weather_tool])

image_url_1 = ""
image_url_2 = ""

message = HumanMessage(content=[
    {"type": "text", "text": "请描述一下这两张图片的天气"},
    {"type": "image_url", "image_url": {"url": image_url_1}},
    {"type": "image_url", "image_url": {"url": image_url_2}}
])

response = model_with_tools.invoke([message])
print(response.tool_calls)
