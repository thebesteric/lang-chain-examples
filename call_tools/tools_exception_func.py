from langchain_core.tools import ToolException, StructuredTool


def get_weather(city: str) -> str:
    """获取天气的工具"""
    if city == "合肥":
        return "天气晴朗"
    # ToolException 为异常处理工具类
    raise ToolException(f"错误，没有名为 {city} 的城市")

def _handle_error(e: ToolException) -> str:
    """异常错误处理"""
    # 可以自定义异常返回的格式，然后根据规则判断如何处理
    return f"工具调用时期发生如下错误：{e.args[0]}"


get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    # 利用函数直接处理，可作为全局异常处理类
    handle_tool_error=_handle_error,
)

response = get_weather_tool.invoke({"city": "北京"})
print(response)