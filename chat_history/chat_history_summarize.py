from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough

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

llm = Ollama(model='llama3.1:8b', temperature=0.0)
chain = prompt | llm

chain_with_message_history = RunnableWithMessageHistory(
    runnable=chain,
    # 获取历史消息
    get_session_history=lambda session_id: temp_chat_histories,
    # 用户输入的消息
    input_messages_key="input",
    # 历史消息
    history_messages_key="chat_history"
)


def summarize_messages(chain_input) -> bool:
    stored_messages = temp_chat_histories.messages
    if len(stored_messages) == 0:
        return False
    summarize_prompt = ChatPromptTemplate.from_messages(
        [
            # 占位符，提供历史消息
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "将上述聊天记录总结成一条摘要消息，尽可能包含更多的具体细节。"),
        ]
    )
    summarize_chain = summarize_prompt | llm
    summary_message = summarize_chain.invoke({"chat_history": stored_messages})
    temp_chat_histories.clear()
    temp_chat_histories.add_ai_message(summary_message)
    return True


chain_with_summarizing = (
        RunnablePassthrough.assign(messages_summarized=summarize_messages)
        | chain_with_message_history
)

response = chain_with_summarizing.invoke(
    {"input": "你回答：我的名字？下午在干嘛？心情如何？"},
    {"configurable": {"session_id": "unused"}},
)
print(response)

for message in temp_chat_histories.messages:
    print(f"after: [{message.type}] {message.content}")
