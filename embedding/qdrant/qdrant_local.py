from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_text_splitters import CharacterTextSplitter

# pip install --upgrade --quiet  langchain-qdrant langchain-openai langchain
# pip install langchain-qdrant

loader = TextLoader("../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 获取 embed_model 路径
embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-m3")
# embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-large-zh-v1.5")
embed_model_absolute_path = str(embed_model_relative_path.resolve())
# 创建 embed 模型
embed_model = HuggingFaceEmbeddings(model_name=embed_model_absolute_path)

qdrant = Qdrant.from_documents(
    docs,
    embed_model,
    location=":memory:",  # 本地模式，仅内存存储
    collection_name="my_documents",
)

query = "Pixar公司是做什么的?"
found_docs = qdrant.similarity_search(query)
print(found_docs[0].page_content)
