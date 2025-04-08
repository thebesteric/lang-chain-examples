from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

embeddings = OpenAIEmbeddings()
chroma_client = chromadb.EphemeralClient()

# 加载文档
loader = TextLoader("../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()

# 将其分割成片段
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

openai_lc_client = Chroma.from_documents(
    docs, embeddings, client=chroma_client, collection_name="openai_collection"
)
query = "Pixar公司是做什么的?"
docs = openai_lc_client.similarity_search(query)
print(docs[0].page_content)
