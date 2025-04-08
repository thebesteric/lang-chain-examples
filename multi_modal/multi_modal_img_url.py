from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage

image_url = "https://vcg02.cfp.cn/creative/vcg/800/new/VCG211229495808.jpg"

llm = Ollama(model='deepseek-r1:8b', temperature=0.0)

message = HumanMessage(content=[
    {"type": "text", "text": "请描述一下这张图片的内容"},
    {
        "type": "image_url",
        # 如果图片地址由于网络问题，导致大模型访问不到，则可能出现问题
        "image_url": {
            "url": image_url
        }
    }
])

response = llm.invoke([message])
print(response)
