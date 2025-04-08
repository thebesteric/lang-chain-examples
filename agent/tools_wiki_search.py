from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import quote

from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 获取 embed_model 路径
embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-m3")
embed_model_absolute_path = str(embed_model_relative_path.resolve())
print(f"embed_model_absolute_path: {embed_model_absolute_path}")
# 创建 embed 模型
embed_model = HuggingFaceEmbeddings(model_name=embed_model_absolute_path)

# 将网址进行 URLEncode
query_str = "猫"
encode_query_str = quote(query_str)

loader = WebBaseLoader(f"https://zh.wikipedia.org/wiki/{encode_query_str}")
docs = loader.load()

# 将文档进行切割
documents = RecursiveCharacterTextSplitter(
    # chunk_size 参数在 RecursiveCharacterTextSplitter 中用于指定每个文档块的最大字符数。
    # chunk_overlap 参数用于指定每个文档块之间的重叠字符数。这意味着，当文档被拆分成我小的块时，每个块的未尾部分会与下一个块的开头部分的重叠数量。
    # 如：第一个块包含字符 1 到 1000。第二个块包含字符 801 到 1800。第三个块包含字符 1601 到 2600。
    chunk_size=1000,
    chunk_overlap=200,
).split_documents(docs)

# 将文档进行向量化
vector = FAISS.from_documents(documents, embed_model)
# 获取检索器
retriever = vector.as_retriever()

# print(retriever.invoke("猫的特征")[0])

# 创建检索器工具
wiki_retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="wiki_search",
    description="搜索维基百科",
)