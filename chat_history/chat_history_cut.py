from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_ollama import ChatOllama

temp_chat_histories = ChatMessageHistory()

temp_chat_histories.add_user_message("我叫Jack，你好")
temp_chat_histories.add_ai_message("你好")
temp_chat_histories.add_user_message("我今天心情很好")
temp_chat_histories.add_ai_message("你今天心情怎么样")
temp_chat_histories.add_user_message("我下午在打篮球")
temp_chat_histories.add_ai_message("你下午在做什么")

for message in temp_chat_histories.messages:
    print(f"before: [{message.type}] {message.content}")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手，尽量回答所有问题. 提供的聊天历史包括与您交谈的用户的先前的消息。"),
    # 占位符，提供历史消息
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

llm = ChatOllama(model='llama3.1:8b', temperature=0.0)
chain = prompt | llm


def trim_messages(chain_input) -> bool:
    """
    对历史消息进行裁剪
    :param chain_input:
    :return:
    """
    stored_messages = temp_chat_histories.messages
    if len(stored_messages) <= 2:
        return False
    temp_chat_histories.clear()
    # 只保留最新的两条消息
    for message in stored_messages[-2:]:
        temp_chat_histories.add_message(message)
    return True


chain_with_message_history = RunnableWithMessageHistory(
    runnable=chain,
    # 获取历史消息
    get_session_history=lambda session_id: temp_chat_histories,
    # 用户输入的消息
    input_messages_key="input",
    # 历史消息
    history_messages_key="chat_history"
)

chain_with_trimming = (
        RunnablePassthrough.assign(messages_trimmed=trim_messages)
        | chain_with_message_history
)

response = chain_with_trimming.invoke(
    # {"input": "我叫什么名字"},
    {"input": "你知道今天下午我在做什么？"},
    {"configurable": {"session_id": "unused"}},
)

print(response)

for message in temp_chat_histories.messages:
    print(f"after: [{message.type}] {message.content}")
