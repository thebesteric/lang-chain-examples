import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

# 项目运行方式：streamlit run rag_chat.py

# 设置标题
st.set_page_config(page_title="企业文档问答系统", page_icon=":robot:", layout="wide")
st.title("企业文档问答系统")

# 侧边栏文件上传
uploaded_files = st.sidebar.file_uploader(
    label="上传文档",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.warning("请先上传文档")
    st.stop()


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # 读取上传的文档，并写入临时目录
    docs = []
    temp_dir = tempfile.TemporaryDirectory(dir="./temp")
    print("temp_dir:", temp_dir)
    print("temp_file_dir:", tempfile.gettempdir())
    print("uploaded_files:", uploaded_files)
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # 加载文本文件
        loader = TextLoader(temp_file_path, encoding="utf-8")
        docs.extend(loader.load())

    # 进行文档分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    print(f"Number of split documents: {len(split_docs)}")

    # 加载本地向量模型
    # embed_model_relative_path = Path(f"/Users/wangweijun/AI/models/BAAI/bge-m3")
    # embed_model_absolute_path = str(embed_model_relative_path.resolve())
    # print(f"embed_model_absolute_path: {embed_model_absolute_path}")
    # embeddings = HuggingFaceEmbeddings(model_name=embed_model_absolute_path)

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    print("Embeddings model loaded successfully!")



    # 文档向量化，返回向量数据库
    vector_db = Chroma.from_documents(split_docs, embeddings)

    # 创建文档检索器
    retriever = vector_db.as_retriever()
    return retriever


# 获取
retriever = configure_retriever(uploaded_files)

# 如果 session_state 中没有历史记录或者用户点击了重置按钮，则清空历史记录
if "messages" not in st.session_state or st.sidebar.button("重置对话"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好，我是你的问答助手，你可以向我提问文档中的任何问题。"}]

# 加载历史聊天记录
for msg in st.session_state.messages:
    print(f"load message: {msg}")
    print(f"role: {msg['role']}, content: {msg['content']}")
    st.chat_message(msg["role"]).write(msg["content"])

# 创建检索工具
tool = create_retriever_tool(
    retriever=retriever,
    name="文档检索",
    description="用于搜索文档内容"
)

tools = [tool]

# 创建聊天历史记录
msgs = StreamlitChatMessageHistory(key="messages")
# 创建对话缓冲区内存
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)

# 指令模板
instructions = """
你是一个设计用于查询文档来回答用户问题的人工智能助手。
你可以使用文档检索工具，并基于检索到的内容来回答用户的问题。
你可能不需要查询文档就知道答案，但你仍应该使用文档检索工具来查询文档获取文档中的答案。
如果你从文档中找不到任何信息，不要尝试自己回答问题，只需要返回”抱歉，我没有找到相关信息。”作为答案返回给用户即可。
"""

base_prompt_template = """
{instructions}

TOOLS:
------

You have access to following tools:

{tools}

To user a tool, please use the following format:

\u200D```
Thought: Do I need to call a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: {input}
Observation: the result of the action
\u200D```

When you have a final answer to the Human, or if you do not need to use a tool, you MUST use the format::

\u200D```
Thought: Do I need to call a tool? No
Final Answer: [Your response here]
```
\u200D

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

# 创建基础提示模板
base_prompt = PromptTemplate.from_template(base_prompt_template)

# 创建部分填充的提示模板
prompt = base_prompt.partial(instructions=instructions)

llm = OllamaLLM(model="qwen2.5:7b", temperature=0.0)
# llm = ChatOpenAI(
#     model='qwen2.5:7b',
#     temperature=0.0,
#     # base_url 必须符合 OpenAI 格式，包含 V1
#     base_url="http://localhost:11434/v1",
#     # api_key 随意输入
#     api_key=SecretStr("ollama"),
# )


# 创建 React Agent
agent = create_react_agent(llm, tools, prompt)
# 创建 agent 执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors="没有从知识库中检索到内容",
)

# 创建聊天输入框
user_input = st.chat_input(placeholder="请开始提问吧！")

# 如果用户输入了消息
if user_input:
    print(f"user_input: {user_input}")
    # 将用户输入添加到聊天历史记录
    st.session_state.messages.append({"role": "user", "content": user_input})
    # 显示用户消息
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        # 创建 streamlit 回调处理器
        st_callback = StreamlitCallbackHandler(st.container())
        # Agent 执行过程中的日志回调显示在 streamlit container 中（如思考，选择工具，执行查询，返回结果等）
        config = {"callbacks": [st_callback]}
        # 调用 agent 执行器，返回结果
        response = agent_executor.invoke({"input": user_input}, config=config)
        # 添加助手消息到 session_state
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        # 显示助手消息
        st.write(response["output"])