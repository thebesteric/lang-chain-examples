import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Zilliz
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# pip install --upgrade --quiet pymilvus

# https://cloud.zilliz.com

loader = TextLoader("../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
# embed_model = OpenAIEmbeddings()

# 获取 embed_model 路径
embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-m3")
# embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-large-zh-v1.5")
embed_model_absolute_path = str(embed_model_relative_path.resolve())
# 创建 embed 模型
embed_model = HuggingFaceEmbeddings(model_name=embed_model_absolute_path)

vector_db = Zilliz.from_documents(  # or Milvus.from_documents
    docs,
    embed_model,
    # 存储到collection_1中
    collection_name="collection_1",
    connection_args={"uri": os.getenv("ZILLIZ_ENDPOINT"), "token": os.getenv("ZILLIZ_TOKEN")},
    # drop_old=True,  # Drop the old Milvus collection if it exists
    auto_id=True,
)

query = "Pixar公司是做什么的?"
docs = vector_db.similarity_search(query)

print(docs[0].page_content)