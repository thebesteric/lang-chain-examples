from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel, SecretStr
from langchain_ollama import OllamaLLM, ChatOllama

# 需要安装 defusedxml
# pip install defusedxml


llm = ChatOpenAI(model='qwen2.5:7b', base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))


# 创建一个 XML 格式输出解析器
# xml_parser = XMLOutputParser()
# 自定义 xml 标签
xml_parser = XMLOutputParser(tags=["movies", "actor", "film", "name", "genre", "year"])

print("============== 格式说明书 ==============")
print(xml_parser.get_format_instructions())
print("======================================")

prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}",
    input_variables=["query"],
    # 相当于格式说明书
    partial_variables={"format_instructions": xml_parser.get_format_instructions()},
)

chain = prompt | llm

response = chain.invoke({"query": "生成周星驰的电影作品列表，按照拍摄时间倒序"})
print(response.content)

# xml_response = xml_parser.parse(response.content)
# print(xml_response)

# # 流式输出
# for s in chain.stream({"query": "生成周星驰的电影作品列表，按照拍摄时间倒序"}):
#     print(s)