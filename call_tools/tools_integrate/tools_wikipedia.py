from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from wikidata.client import Client


# pip install --upgrade --quiet  wikipedia
# pip install wikidata

# 创建 wiki_client 对象
wiki_client = Client()
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500, wiki_client=wiki_client)

tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

# Page: Largest and heaviest animals
# Summary: The largest animal currently alive is the blue whale. The maximum recorded weight was 190 tonnes (209 US tons) for a specimen measuring 27.6 metres (91 ft), whereas longer ones, up to 33 metres (108 ft), have been recorded but not weighed. It is estimated that this individual could have a mass of 250 tonnes or more. The longest non-colonial animal is the lion's mane jellyfish (37 m, 120 ft).
# In 2023, paleontologists estimated that the extinct whale Per
print(tool.invoke({"query": "what is biggest animal in the world?"}))

# Name: wikipedia
print(f"Name: {tool.name}")
# Description: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.
print(f"Description: {tool.description}")
# Args schema: {'query': {'description': 'query to look up on wikipedia', 'title': 'Query', 'type': 'string'}}
print(f"Args schema: {tool.args}")
# Returns directly: False
print(f"Returns directly: {tool.return_direct}")