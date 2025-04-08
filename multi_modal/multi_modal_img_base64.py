import base64
import httpx
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

image_url = "https://vcg02.cfp.cn/creative/vcg/800/new/VCG211229495808.jpg"
# 将图片转化为 base64 编码
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

# 需要模型有视觉支持，如 deepseek-ai/deepseek-vl2，Qwen/Qwen2.5-VL-7B-Instruct
llm = ChatOllama(model='deepseek-r1:8b', temperature=0.0)

message = HumanMessage(content=[
    {"type": "text", "text": "请描述一下这张图片的内容"},
    {
        "type": "image_url",
        # 传递 base64 编码的图片
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_data}"
        }
    }
])

response = llm.invoke([message])
print(response)
