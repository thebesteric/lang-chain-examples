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

# 创建简单的 ids
ids = [str(i) for i in range(1, len(docs) + 1)]
# 添加数据
example_db = Chroma.from_documents(documents=docs, embedding=embed_model, ids=ids, collection_name="langchain")

query = "Pixar公司是做什么的?"
docs = example_db.similarity_search(query)

print("更新前内容：")
print(example_db._collection.get(ids=[ids[0]]))

# 更新文档的元数据
docs[0].metadata = {
    "source": "../resource/knowledge.txt",
    "new_value": "hello world",
}
# 执行更新
example_db.update_document(ids[0], docs[0])
print("更新后内容：")
print(example_db._collection.get(ids=[ids[0]]))

# 删除最后一个文档
print("删除前计数", example_db._collection.count())
print(example_db._collection.get(ids=[ids[-1]]))

# 执行删除操作
example_db._collection.delete(ids=[ids[-1]])
print("删除后计数", example_db._collection.count())
print(example_db._collection.get(ids=[ids[-1]]))
