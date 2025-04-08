from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel, SecretStr
from langchain_ollama import OllamaLLM, ChatOllama


# llm = ChatOllama(model='qwen2.5:7b', temperature=0.0)
llm = ChatOpenAI(model='qwen2.5:7b', base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))


# 定义期望的结构化数据格式
class Joke(BaseModel):
    # description 属性：会告诉大模型做语义理解的时候，加入这些条件，也就是让大模型生成一个固定格式的笑话
    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的答案")


# 创建一个 JSON 格式输出解析器，并生成一个指定格式的 JSON 对象
json_parser = JsonOutputParser(pydantic_object=Joke)

print("============== 格式说明书 ==============")
print(json_parser.get_format_instructions())
print("======================================")

prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}",
    input_variables=["query"],
    # 相当于格式说明书
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)

chain = prompt | llm | json_parser

response = chain.invoke({"query": "告诉我一个笑话"})
print(response)

# print(json_parser.parse(response))

# # 流式输出
# for s in chain.stream({"query": "告诉我一个笑话"}):
#     print(s)
