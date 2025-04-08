import os
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 导入 langgraph 检查点内存保存器，用于保存状态
from langgraph.checkpoint.memory import MemorySaver
# 导入 langgraph 状态图和消息图
from langgraph.graph import END, StateGraph, MessageGraph, MessagesState
# 导入 langgraph 工具节点，主要用来调用工具
from langgraph.prebuilt import ToolNode

os.environ['LANGSMITH_PROJECT'] = 'langgraph_demo'


# pip install -U langgraph

# 定义一个天气搜素工具
@tool(description="天气搜索工具")
def weather_search_tool(query: str):
    """天气搜索工具"""
    print(f"===> call tools: weather_search_tool, query: {query}")
    if "上海" in query:
        return "天气很热，37度"
    if "北京" in query:
        return "天气很冷，零下10度"
    if "合肥" in query:
        return "天气凉爽，23度"
    return "未知城市"

# 定义一个位置搜素工具
@tool(description="地理位置搜索工具")
def position_search_tool(query: str):
    """地理位置搜索工具"""
    print(f"===> call tools: position_search_tool, query: {query}")
    if "上海" in query:
        return "上海位于中国华东"
    if "北京" in query:
        return "北京位于中国华北"
    if "合肥" in query:
        return "合肥位于中国东北"
    return "未知城市"


# 将工具加入工具列表
tools = [weather_search_tool, position_search_tool]

# 创建工具节点（包含工具列表）
# 所有的工具都要封装成一个工具节点
tool_node = ToolNode(tools)

# 1、初始化模型模型和工具，并将工具绑定到模型
llm = ChatOpenAI(model="qwen2.5:7b", temperature=0.0, api_key=SecretStr("ollama"), base_url="http://localhost:11434/v1")
# 给模型绑定工具集
# 注意：必须返回 llm，否则不会生效
llm_with_tools = llm.bind_tools(tools)

# 定义路由函数
def should_continue(state: MessagesState) -> Literal["tools", END]:
    """判断是否需要继续（路由函数）"""
    messages = state['messages']
    # 获取最后一条消息
    last_message = messages[-1]
    # 如果 llm 调用了工具，表示大模型需要使用工具，则转到 tools 节点
    if last_message.tool_calls:
        # 返回节点的名字
        return "tools"
    # 否则，停止（回复用户）
    return END

# 调用大模型的函数
def call_llm(state: MessagesState):
    """调用模型"""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    # 返回状态
    return {"messages": [response]}


# 2、定义一个新的状态图
workflow = StateGraph(MessagesState)

# 3、添加图节点，一个代理节点和一个工具节点
workflow.add_node("agent", call_llm)
workflow.add_node("tools", tool_node)

# 4、定义入口节点
# 入口节点为 agent，这意味着 agent 节点是第一个被调用的节点
# 即：__start__ ==> agent
workflow.set_entry_point("agent")

# 5、添加条件边
# 即：
# 第一个条件：agent --> tools
# 第二个条件：agent --> __end__
workflow.add_conditional_edges(
    # 首先定义起始节点，使用 agent
    # 这意味着这些边是在调用 agent 节点后采用的
    source="agent",
    # 接下来，决定要采用哪条边（下个节点的调用函数）
    path=should_continue
)

# 6、添加普通边
# 添加从 tools 到 agent 的普通边
# 这意味着当 tools 节点被调用后，会再次调用 agent 节点
# 即：tools ==> agent
workflow.add_edge("tools", "agent")

# 初始化内存以在图运行之间持久化状态
# 可以存放在 redis 中
checkpointer = MemorySaver()

# 7、编译图
# 编译成 langchain 可以运行的对象，意味着你可以像其他可运行对象一样被使用
# checkpointer 非必须
app = workflow.compile(checkpointer=checkpointer)

# 8、执行图，调用可运行对象
final_state = app.invoke(
    input={"messages": [HumanMessage(content="上海的天气如何？北京的地理位置在哪里？")]},
    # 必须是 thread_id
    config={"configurable": {"thread_id": 123}}
)

# 9、打印结果
# 从 final_state 中获取最后一条消息的内容
response = final_state["messages"][-1].content
print(response)

# 验证存储点是否生效，再次执行图，调用可运行对象
final_state = app.invoke(
    input={"messages": [HumanMessage(content="我刚才问的什么问题？")]},
    config={"configurable": {"thread_id": 123}}
)
response = final_state["messages"][-1].content
print(response)

# 将图绘制成图片保存
graph_image_filename = "langgraph_hello.png"
if not os.path.exists(graph_image_filename):
    print(f"生成 graph 示例图: {graph_image_filename}")
    graph = app.get_graph().draw_mermaid_png()
    with open(graph_image_filename, "wb") as f:
        f.write(graph)
