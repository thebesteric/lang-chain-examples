import asyncio

from langchain_core.tools import StructuredTool

"""
使用 StructuredTool.from_function 创建工具，支持同步和异步函数
提供了以下参数：
    func: 指定一个同步函数，当在同步上下文中调用工具时，将使用此函数
    coroutine: 指定一个异步函数，当在异步上下文中调用工具时，将使用此函数
"""

# 同步方法
def multiply(a: int, b: int) -> int:
    """这是一个用来进行两个数字相乘的工具"""
    return a * b


# 异步方法
async def async_multiply(a: int, b: int) -> int:
    """这是一个用来进行两个数字相乘的工具"""
    return a * b


async def main():
    # func: 指定一个同步函数，当在同步上下文中调用工具时，将使用此函数
    # coroutine: 指定一个异步函数，当在异步上下文中调用工具时，将使用此函数
    calculator = StructuredTool.from_function(func=multiply, coroutine=async_multiply)
    # invoke: 同步调用，会调用 multiply
    print("同步调用结果：", calculator.invoke({"a": 3, "b": 4}))
    # ainvoke: 异步调用，会调用 async_multiply
    print("异步调用结果：", await calculator.ainvoke({"a": 4, "b": 5}))


asyncio.run(main())
