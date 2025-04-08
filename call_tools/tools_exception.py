from langchain_core.tools import ToolException, StructuredTool


def get_weather(city: str) -> str:
    """获取天气的工具"""
    if city == "合肥":
        return "天气晴朗"
    # ToolException 为异常处理工具类
    raise ToolException(f"错误，没有名为 {city} 的城市")


get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    # 如果 handle_tool_error 设置为 true，则返回 ToolException 的异常文本，False 则会抛出 ToolException 异常
    # 默认为 False，就会抛出 ToolException 异常
    # 如果设置异常文本，则会返回该文本
    handle_tool_error=True,
    # handle_tool_error="没有找到对应的城市天气" # 直接使用错误描述信息（不推荐）
)

response = get_weather_tool.invoke({"city": "北京"})
print(response)