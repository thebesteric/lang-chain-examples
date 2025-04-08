from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 创建一个 Chroma 客户端
persistent_client = chromadb.PersistentClient(path="./chroma_db")

# 使用客户端创建一个集合
# 默认会下载：all-MiniLM-L6-v2
my_collection = persistent_client.get_or_create_collection("my_collection_name")
# 向集合中添加数据，并指定 ID
my_collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

langchain_chroma = Chroma(
    client=persistent_client,
    collection_name="my_collection_name",
)
print("在集合中有", langchain_chroma._collection.count(), "个文档")
