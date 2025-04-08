import os

from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 实例化一个 ChatOpenAI 对象
# os.environ["OPENAI_API_KEY"] = "sk-xxx"
# llm = ChatOpenAI()

# llm = OllamaLLM(model='llama3.1:8b', temperature=0.0, base_url="http://localhost:11434")
llm = ChatOpenAI(model='llama3.1:8b', temperature=0.0, base_url="http://localhost:11434/v1", api_key=SecretStr("ollama"), )

# 创建一个聊天提示词模板，定义系统和用户的消息角色
# 系统角色设定为世界级的AI技术专家，以提升生成内容的质量
# 用户角色使用占位符，以便在调用时传入具体的问题或请求
# 角色只能是：'human', 'user', 'ai', 'assistant', or 'system'
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是世界级的AI技术专家"),
    ("user", "{input}"),
])

# 定义一个字符串输出解析器，用于将AI模型生成的结果解析为字符串类型
# 对于有些模型，会输出一些 token 信息，使用 StrOutputParser 可以将多余的内容去掉，只保留 content
str_out_put_parser = StrOutputParser()

# 构建一个链式处理流程，将 聊天提示模板 与 llm 对象和 str_out_put_parser 结合
# 即：prompt -> llm -> str_out_put_parser
# 这样可以将用户输入经过模板格式化后，直接传递给AI模型进行处理
chain = prompt | llm | str_out_put_parser

# 使用链式处理流程生成关于AI的技术文章
# 传入具体的问题或请求，即希望AI模型生成的文章主题和长度要求
result = chain.invoke({"input": "帮我写一篇关于AI的技术文章，100个字"})
# 输出生成的文章内容
print(result)

