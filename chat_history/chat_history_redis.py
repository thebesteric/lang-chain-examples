from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
# 引入会话配置字段
from langchain_core.runnables import ConfigurableFieldSpec
# 引入 Redis 存储历史消息类
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_ollama import ChatOllama

# 创建一个聊天提示词模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个非常有用的助手，擅长处理 {ability} 问题. 给我的答案控制在20字以内"),
    # 历史消息占位符
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

llm = ChatOllama(model='llama3.1:8b', temperature=0.0)
runnable_chain = prompt | llm

# Redis 配置
REDIS_URL = "redis://localhost:6379/0"


def get_redis_message_history(user_id: str, conversation_id: str) -> RedisChatMessageHistory:
    session_id = f"{user_id}:{conversation_id}"
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix="llm_chat_history:",
        ttl=300
    )


# 创建一个带历史记录的 runnable 链
with_message_history = RunnableWithMessageHistory(
    runnable_chain,
    get_redis_message_history,
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
        "ability": "math",
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

# 再次启动程序，执行如下对话，会返回正确的响应
# response = with_message_history.invoke(
#     {
#         "ability": "数学",
#         "input": "在描述一下",
#     },
#     config={"configurable": {"user_id": "u-123456", "conversation_id": "c-123456"}},
# )
# # 余弦是三角学中的一个基本概念，表示两个角之间的正弦值。
# print("user_id: u-123456, conversation_id: c-123456\n", response)
