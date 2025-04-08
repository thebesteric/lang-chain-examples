from langchain.output_parsers import YamlOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import Field, BaseModel
from langchain_ollama import OllamaLLM, ChatOllama

# llm = OllamaLLM(model='qwen2.5:7b', temperature=0.0)
llm = ChatOllama(model='qwen2.5:7b', temperature=0.0)

# 定义期望的结构化数据格式
class Joke(BaseModel):
    # description：会告诉大模型做语义理解的时候，加入这些条件，也就是让大模型生成一个固定格式的笑话
    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的答案")

# 创建一个 JSON 格式输出解析器，并生成一个指定格式的 JSON 对象
yaml_parser = YamlOutputParser(pydantic_object=Joke)

print("============== 格式说明书 ==============")
print(yaml_parser.get_format_instructions())
print("======================================")

prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}",
    input_variables=["query"],
    # 相当于格式说明书
    partial_variables={"format_instructions": yaml_parser.get_format_instructions()},
)

chain = prompt | llm

response = chain.invoke({"query": "告诉我一个笑话"})
print(response)

# print(yaml_parser.parse(response))

# # 流式输出
# for s in chain.stream({"query": "告诉我一个笑话"}):
#     print(s)