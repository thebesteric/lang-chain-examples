from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, \
    SystemMessagePromptTemplate, MessagesPlaceholder

print("# =================================")
print("# PromptTemplate：创建一个简单的模板，其中包含两个变量 topic 和 adjective")
print("# 适用于：简单的任务场景")
print("# =================================")

# 创建一个简单的模板，其中包含两个变量 topic 和 adjective
simple_template = PromptTemplate.from_template(
    "给我讲一个关于{topic}的笑话，其中要包括{adjective}"
)

# 使用 format 方法替换模板变量，返回一行字符串
result = simple_template.format(topic="猫", adjective="老鼠")
print(result)
# 给我讲一个关于猫的笑话，其中要包括老鼠


print("\n\n")
print("# =================================")
print("# 创建一个聊天的模板")
print("# 适用于：聊天场景，记住上下文的聊天记录")
print("# =================================")

# 通过一个消息数组创建聊天消息模板
# 数组每一个元素代表一条消息，每个消息元组，第一个元素代表消息角色（也成为消息类型），第二个元素代表消息内容。
# 消息角色：system 代表系统消息、human 代表人类消息，ai 代表 LLM 返回的消息内容
# 下面消息定义了 2 个模板参数 name 和 user_input

# chat_template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template(template="你是一位人工智能助手，你的名字是{name}"),
#         HumanMessage(content="你好"),
#         AIMessage(content="我很好，谢谢！"),
#         HumanMessagePromptTemplate.from_template(template="{user_input}"),
#     ]
# )

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位人工智能助手，你的名字是{name}"),
        ("human", "你好"),
        ("ai", "我很好，谢谢！"),
        ("human", "{user_input}"),
    ]
)

# 通过模板参数格式化模板内容
messages = chat_template.format_messages(name="Bob", user_input="你叫什么名字")
print(messages)
# [SystemMessage(content='你是一位人工智能助手，你的名字是Bob', additional_kwargs={}, response_metadata={}),
# HumanMessage(content='你好', additional_kwargs={}, response_metadata={}),
# AIMessage(content='我很好，谢谢！', additional_kwargs={}, response_metadata={}),
# HumanMessage(content='你叫什么名字', additional_kwargs={}, response_metadata={})]

print("\n\n")
print("# =================================")
print("# MessagesPlaceholder：这个提示模板负责在特定位置添加消息列表")
print("# 提示词中包含交互样本的作用是为了帮助模型更好地理解用户的意图")
print("# 小样本提回 示模板是指使用一组少量的示例来指导模型处理新的输入")
print("# 从而更好地回答问题或执行任务。。这些示例可以用来训练模型，以便模型可以更好地理解和回答类似的问题")
print("# 适用于：聊天场景，记住上下文的聊天记录，可以更具需要插入不同类型的消息列表，让模型更加理解用户的意图，回答更加准确")
print("# =================================")

placeholder_template = ChatPromptTemplate.from_messages([
    ("system", "你是一位人工智能助手，你的名字是{name}"),
    # 占位符，在模板中插入消息列表
    MessagesPlaceholder("msgs")
])
result = placeholder_template.invoke({
    "name": "Bob",
    "msgs": [
        ("human", "你好"),
        ("ai", "我很好，谢谢！"),
        ("human", "你叫什么名字"),
    ]
})
print(result)
# [SystemMessage(content='你是一位人工智能助手，你的名字是Bob', additional_kwargs={}, response_metadata={}),
# HumanMessage(content='你好', additional_kwargs={}, response_metadata={}),
# AIMessage(content='我很好，谢谢！', additional_kwargs={}, response_metadata={}),
# HumanMessage(content='你叫什么名字', additional_kwargs={}, response_metadata={})]
