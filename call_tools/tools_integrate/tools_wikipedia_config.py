from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field
from wikidata.client import Client


# pip install --upgrade --quiet  wikipedia
# pip install wikidata

class WikiInputs(BaseModel):
    """维基百科工具的输入类"""
    query: str = Field(title="查询字符串", description="维基百科查询的查询字符串")


# 创建 wiki_client 对象
wiki_client = Client()
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500, wiki_client=wiki_client)

tool = WikipediaQueryRun(
    name="wiki-tool",
    description="维基百科查询工具，输入查询字符串，返回查询结果",
    args_schema=WikiInputs,
    api_wrapper=wiki_api_wrapper,
    return_direct=True
)

print(tool.invoke({"query": "langchain"}))

# Name: wiki-tool
print(f"Name: {tool.name}")
# Description: 维基百科查询工具，输入查询字符串，返回查询结果
print(f"Description: {tool.description}")
# Args schema: {'query': {'description': '维基百科查询的查询字符串', 'title': '查询字符串', 'type': 'string'}}
print(f"Args schema: {tool.args}")
# Returns directly: True
print(f"Returns directly: {tool.return_direct}")