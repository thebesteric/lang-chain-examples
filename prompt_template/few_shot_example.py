# 定义少样本示例
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

# 创建一个 PromptTemplate 对象，用于生成每个示例的提示词
example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题: {question}\n{answer}")

print("# =================================")
print("# 创建小样本示例的格式化程序")
print("# =================================")
# 提取 examples 示例集合的一个示例的内容，用于格式化模板内容
# examples[0] = ｛'question'：'乔治•华盛顿的祖父母中的母亲是谁？'，'answer'：'Joseph Ball'｝
# **examples[0] = question=乔治•华盛顿的祖父母中的母亲是谁？，answer=Joseph Ball
print(example_prompt.format(**examples[0]))

print("\n\n")
print("# =================================")
print("# 将所有示例提供给程序")
print("# =================================")
# 创建一个 FewShotPromptTemplate 对象，用于生成少样本示例的提示词
# 提示词中包含多个示例，每个示例都包含问题（question）和答案（answer）
# 提示词中包含一个通用的提示，用于生成新问题的答案
few_shot_prompt = FewShotPromptTemplate(
    # 传入所有示例
    examples=examples,
    # 模板格式
    example_prompt=example_prompt,
    # 示例之间的分隔符
    example_separator="\n\n",
    # 问题格式
    suffix="问题: {input}",
    # 占位符，非必须
    # input_variables=["input"],
)

#
print(few_shot_prompt.format(input="乔治华盛顿的父亲是谁？"))