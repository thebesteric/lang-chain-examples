from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 引入聊天消息历史记录
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
# 引入会话配置字段
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_ollama import ChatOllama

# 创建一个聊天提示词模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个非常有用的助手，擅长处理 {ability} 问题. 给我的答案控制在20字以内"),
    # 历史消息占位符
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

llm = ChatOllama(model='llama3.1:8b', temperature=0.0)
runnable_chain = prompt | llm | StrOutputParser()

# 用来存储会话的历史记录
store = {}


def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    """
    获取会话历史记录
    :param user_id: 用户 ID
    :param conversation_id: 会话 ID
    :return: 对应会话的历史记录
    """
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    # 取出历史记录
    return store[(user_id, conversation_id)]


# 创建一个带历史记录的 runnable 链
with_message_history = RunnableWithMessageHistory(
    runnable_chain,
    # 获取会话历史记录的函数
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    history_factory_config=[
        ConfigurableFieldSpec(
            # 对应 get_session_history 的参数名：user_id
            id="user_id",
            annotation=str,
            name="User ID",
            description="用户唯一标识",
            default="",
            is_shared=True
        ),
        ConfigurableFieldSpec(
            # 对应 get_session_history 的参数名：conversation_id
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="会话的唯一标识",
            default="",
            is_shared=True
        ),
        # 可以通过 ConfigurableFieldSpec 拓展更多的参数
    ]
)

response = with_message_history.invoke(
    {
        "ability": "数学",
        "input": "余弦是什么意思",
    },
    config={"configurable": {"user_id": "u-123456", "conversation_id": "c-123456"}},
)
# 余弦是三角学中的一个基本概念，指的是两个角之间的正弦值之积。
print("user_id: u-123456, conversation_id: c-123456\n", response)

response = with_message_history.invoke(
    {
        "ability": "数学",
        "input": "你说什么？",
    },
    config={"configurable": {"user_id": "u-123456", "conversation_id": "c-123456"}},
)
# 我说的是余弦（cosine），它是三角学中一个重要的概念。
print("user_id: u-123456, conversation_id: c-123456\n", response)

response = with_message_history.invoke(
    {
        "ability": "数学",
        "input": "你说什么？",
    },
    config={"configurable": {"user_id": "u-123456", "conversation_id": "c-654321"}},
)
# 我是你的数学助手，可以帮助你解决各种数学问题。请问你需要帮助哪个方面的题目？
print("user_id: u-123456, conversation_id: c-654321\n", response)

print("store 中的信息如下：\n", store)
