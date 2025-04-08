import asyncio

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

"""
使用 StructuredTool.from_function 创建工具，相比较 @tool 装饰器创建工具，更加灵活
"""

# 同步方法
def multiply(a: int, b: int) -> int:
    """这是一个用来进行两个数字相乘的工具"""
    return a * b


# 异步方法
async def async_multiply(a: int, b: int) -> int:
    """这是一个用来进行两个数字相乘的工具"""
    return a * b


class CalculatorInput(BaseModel):
    a: int = Field(title="first", description="第一个数字")
    b: int = Field(title="second", description="第二个数字")


async def main():
    calculator = StructuredTool.from_function(
        # name: 工具的名称，默认为函数的名称
        name="calculator",
        # func: 指定一个同步函数，当在同步上下文中调用工具时，将使用此函数
        func=multiply,
        # coroutine: 指定一个异步函数，当在异步上下文中调用工具时，将使用此函数
        coroutine=async_multiply,
        # args_schema: 参数的格式的描述
        args_schema=CalculatorInput,
        # return_direct:
        # 设置为 True 时，工具执行完后会直接返回结果，而不会将结果再交给链（Chain）或者代理（Agent）做进一步处理；
        # 设置为 False（默认值），工具执行结果会被传递给链或者代理，由它们继续进行后续处理。
        return_direct=True
    )
    # invoke: 同步调用
    print("同步调用结果：", calculator.invoke({"a": 3, "b": 4}))
    # ainvoke: 异步调用
    print("异步调用结果：", await calculator.ainvoke({"a": 4, "b": 5}))


asyncio.run(main())
