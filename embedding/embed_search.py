from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from numpy import dot
from numpy.linalg import norm

# 获取 embed_model 路径
# embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-m3")
# embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-large-zh-v1.5")
# embed_model_absolute_path = str(embed_model_relative_path.resolve())
# 创建 embed 模型
# embed_model = HuggingFaceEmbeddings(model_name=embed_model_absolute_path)
# print(f"embed model load succeed: {embed_model_absolute_path}")

embed_model = OllamaEmbeddings(model="nomic-embed-text:latest")

def get_embedding(text):
    """文本转为向量值"""
    return embed_model.embed_query(text)

def cosine_similarity(vec1, vec2):
    """
    余弦相似度 =  两个坐标点的点积 / 两个向量的范数的乘积
    A(1,2,3) B(4,5,6)
    点积：A.B = 1*4 + 2*5 + 3*6 = 32
    范数：||A|| = √(1² + 2² + 3²) = √14
    范数：||B|| = √(4² + 5² + 6²) = √77
    余弦相似度：cos(A,B) = 32 / (√14 * √77) = 0.99
    """
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def search_documents(query, documents):
    # 生成查询字符串的向量
    query_embedding = get_embedding(query)
    # 生成文档列表的向量
    doc_embeddings = [get_embedding(doc) for doc in documents]
    # 计算每个文档与查询字符串的相似度
    similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in doc_embeddings]
    print(f"similarities: {similarities}")
    # 找到最相似的文档
    most_similar_index = similarities.index(max(similarities))
    # 返回最相似的文档和相似度得分
    return documents[most_similar_index], max(similarities)


if __name__ == '__main__':
    documents = [
        "OpenAI 的 ChatGPT 是一个大语言模型",
        "天空是蓝色的，阳光灿烂",
        "人工智能正在改变世界",
        "Python 是一种流行的编程语言",
    ]

    query = "天空是什么颜色的"

    most_similar_document, similar_score = search_documents(query, documents)
    print(f"最相似的文档：{most_similar_document}")
    print(f"相似度得分：{similar_score}")