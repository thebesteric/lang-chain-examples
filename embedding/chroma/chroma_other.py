from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# 加载文档
loader = TextLoader("../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
# 将其分割成片段
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 创建开源嵌入函数
# 获取 embed_model 路径
embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-m3")
# embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-large-zh-v1.5")
embed_model_absolute_path = str(embed_model_relative_path.resolve())
# 创建 embed 模型
embed_model = HuggingFaceEmbeddings(model_name=embed_model_absolute_path)

# 将其加载到 Chroma 内存中
db = Chroma.from_documents(docs, embed_model)

# 进行查询
query = "Pixar公司是做什么的?"
# 返回的距离分数是余弦距离。因此，分数越低越好。
docs = db.similarity_search_with_score(query)
# 打印结果
print(docs[0])

print("=" * 100)

# 使用 mmr 进行相似性搜索。
retriever = db.as_retriever(search_type="mmr")
result = retriever.invoke(query)
print(result[0])

print("=" * 100)

# 筛选已更新来源的集合
docs = db.get(where={"source": "../resource/knowledge.txt"})
print(docs)
