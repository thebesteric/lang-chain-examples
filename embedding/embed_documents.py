from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

# embed_model = OllamaEmbeddings(model="nomic-embed-text:latest")

# 获取 embed_model 路径
# embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-m3")
embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-large-zh-v1.5")
embed_model_absolute_path = str(embed_model_relative_path.resolve())
print(f"embed_model_absolute_path: {embed_model_absolute_path}")
# 创建 embed 模型
embed_model = HuggingFaceEmbeddings(model_name=embed_model_absolute_path)

embeddings = embed_model.embed_documents(
    [
        "嗨",
        "嗨，你好",
        "你叫什么名字？",
        "我的朋友们叫我World",
        "Hello World！"
    ]
)

# nomic-embed-text:             5（文档数量） 768（纬度）
# BAAI/bge-m3:                  5（文档数量） 1024（纬度）
# BAAI/bge-large-zh-v1.5:       5（文档数量） 1024（纬度）
print(len(embeddings), len(embeddings[0]))

embeded_response = embed_model.embed_query("对话中提到的名字是什么？")
# 查询嵌入的前 5 个值
# [-0.031191935762763023, -0.015782900154590607, -0.0033305685501545668, 0.004198785871267319, 0.008952697739005089]
print(embeded_response[:5])