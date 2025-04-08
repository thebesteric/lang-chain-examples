from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {
        "question": "谁的寿命更长，穆罕默德•阿里还是艾伦•图灵？",
        "answer":
            """
            这里需要跟进问题吗：是的。
            限进：穆罕默德•阿里去世时多大？
            中间答案：穆罕默德•阿里去世时74岁。
            跟进：艾伦•图灵去世时多大？
            中间答案：艾伦•图灵去世时41岁。
            所以最终答案是：穆罕默德•阿里
            """
    },
    {
        "question": "Craigslist的创始人是什么时候出生的？",
        "answer":
            """
            这里需要跟进问题吗：是的。
            限进：Craigslist的创始人是什么时候出生的？
            中间答案：Craigslist的创始人是Craig Newmark。
            跟进：Craig Newmark是什么时候出生的？
            中间答案：Craig Newmark出生于1952年12月6日。
            所以最终答案是：1952年12月6日。
            """
    },
    {
        "question": "乔治•华盛顿的祖父母中的父亲是谁？",
        "answer":
            """
            这里需要跟进问题吗：是的。
            限进：乔治•华盛顿的母亲是谁？
            中间答案：乔治•华盛顿的母亲是Mary Ball Washington。
            跟进：Mary Ball Washington是的父亲是谁？
            中间答案：Mary Ball Washington的父亲是Joseph Ball。
            所以最终答案是：Joseph Ball。
            """
    },
    {
        "question": "《大白鲨》和《皇家赌场》的导演都来自同一个国家吗？",
        "answer":
            """
            这里需要跟进问题吗：是的。
            限进：《大白鲨》的导演来是谁？
            中间答案：《大白鲨》的导演是Steven Spielberg。
            跟进：Steven Spielberg来自哪里？
            中间答案：Steven Spielberg来自美国。
            跟进：《皇家赌场》的导演是谁？
            中间答案：《皇家赌场》的导演是Martin Campbell。
            跟进：Martin Campbell来自哪里？
            中间答案：Martin Campbell来自新西兰。
            所以最终答案是：不是同一个国家。
            """
    }
]
# 初始化 HuggingFaceEmbeddings 模型
MODEL_DIR = "/Users/wangweijun/AI/models"
embed_model_relative_path = Path(f"{MODEL_DIR}/BAAI/bge-m3")
embed_model_absolute_path = str(embed_model_relative_path.resolve())
print(f"embed_model_absolute_path: {embed_model_absolute_path}")
# 创建 embed 模型
embeddings = HuggingFaceEmbeddings(model_name=embed_model_absolute_path)
# embeddings = OpenAIEmbeddings()

# 使用语义相似性示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 这里是示例集
    examples=examples,
    # 转化为向量：这里是嵌入模型，将对话文本转化为向量
    embeddings=embeddings,
    # 存储向量：存储嵌入向量并执行相似性搜索的向量存储类
    vectorstore_cls=Chroma,
    # 这里是要生成的示例数
    k=1
)

question = "乔治•华盛顿的父亲是谁？"
# question = "大白鲨的导演是那个国家的"
# 选择与问题最相似的示例
selected_examples = example_selector.select_examples({"question": question})

# 创建一个 PromptTemplate 对象，用于生成每个示例的提示词
example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题: {question}\\n{answer}")

prompt = FewShotPromptTemplate(
    # examples=examples,
    # 更换为 selected_examples，这样就可以减少样本数量，提高准确率
    examples=selected_examples,
    example_prompt=example_prompt,
    suffix="问题: {input}",
    input_variables=["input"],
    example_separator="\n\n",
)

print(prompt.format(input=question))