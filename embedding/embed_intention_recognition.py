from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

# 导入数据
loader = TextLoader(file_path="./resource/abilities.txt", encoding="utf-8")
docs = loader.load()

# 数据切分
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
print(f"Number of documents: {len(documents)}")

# 获取文本嵌入模型
embed_model = OllamaEmbeddings(model="nomic-embed-text:latest")

# 存储文档到向量数据库
vector = FAISS.from_documents(documents, embed_model)

# 提示词模板
prompt = ChatPromptTemplate.from_template(
"""
根据提供的上下文回答一下问题：
<context>
{context}
</context>
问题：{input}
"""
)

# 加载大模型
llm = ChatOpenAI(model='qwen2.5:7b', base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))

# 创建大模型和提示词的组合的链
document_chain = create_stuff_documents_chain(llm, prompt)

# 创建向量数据库的检索器
retriever = vector.as_retriever()

# 创建检索器与大模型和提示词的组合的链
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 执行
response = retrieval_chain.invoke({"input": "你能帮我哪些事情？"})
print(response)
