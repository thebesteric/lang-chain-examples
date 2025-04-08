from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langserve import RemoteRunnable

# 创建一个远程借口调用
print("===> 使用 LangServe 的 SKD，同步调用：/chat/invoke")
chat = RemoteRunnable("http://localhost:8000/chat/")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个喜欢写故事的助手"),
        ("user", "写一个故事，主题是：{topic}"),
    ]
)
# llm: 表示一个 key 的名称，可自定义
chain = prompt | RunnableMap({
    "llm": chat
})
# topic：表示一个参数的名称
response = chain.invoke({
    "topic": "猫"
})
print(response)


print("===> 使用 LangServe 的 SKD，同步调用：/chat_str_parser/invoke")
chat_str_parser = RemoteRunnable("http://localhost:8000/chat_str_parser/")
chain = prompt | RunnableMap({
    "llm": chat_str_parser
})
response = chain.invoke({
    "topic": "猫"
})
print(response)

print("===> 使用 LangServe 的 SKD，流式调用：/chat_str_parser/invoke")
for chunk in chain.stream({"topic": "猫"}):
    # print(chunk, end="|", flush=True)
    print(chunk['llm'], end="|", flush=True)


print("===> 使用 LangServe 的 SKD，同步调用：/chat_with_prompt/invoke")
chat_with_prompt = RemoteRunnable("http://localhost:8000/chat_with_prompt/")
chain = RunnableMap({
    "llm": chat_with_prompt
})
response = chain.invoke({
    "topic": "狗"
})
print(response)