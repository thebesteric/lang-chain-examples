# 导入装饰器的库
import asyncio

from langchain_core.tools import tool
from pydantic import BaseModel, Field

print("=============== 同步工具 ==================")

"""
使用 @tool 装饰器创建工具
提供了以下参数：
    name: 工具的名称，默认为函数的名称
    args: 参数的格式的描述
    description: 工具的描述
    return_direct:
        设置为 True 时，工具执行完后会直接返回结果，而不会将结果再交给链（Chain）或者代理（Agent）做进一步处理；
        设置为 False（默认值），工具执行结果会被传递给链或者代理，由它们继续进行后续处理。
"""
@tool
def multiply(a: int, b: int) -> int:
    """这是一个用来进行两个数字相乘的工具"""
    return a * b

# 使用 invoke 方法调用
result = multiply.invoke({"a": 3, "b": 4})
print(f"multiply 同步乘法结果: {result}")

# 结果：multiply
print(f"multiply.name = {multiply.name}")
# 结果：这是一个用来进行两个数字相乘的工具
print(f"multiply.description = {multiply.description}")
# 结果：{'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
print(f"multiply.args = {multiply.args}")
# 结果：False，表示是否直接返回结果给代理或者链继续处理
print(f"multiply.return_direct = {multiply.return_direct}")

print("=============== 自定义描述：同步工具 ==================")


class CalculatorInput(BaseModel):
    a: int = Field(title="first", description="第一个数字")
    b: int = Field(title="second", description="第二个数字")


@tool(name_or_callable="custom_multiply_tool", args_schema=CalculatorInput,
      description="这是一个用来进行两个数字相乘的工具", return_direct=True)
def custom_multiply(a: int, b: int) -> int:
    return a * b


# custom_multiply_tool
print(f"custom_multiply.name = {custom_multiply.name}")
# 这是一个用来进行两个数字相乘的工具
print(f"custom_multiply.description = {custom_multiply.description}")
# {'a': {'description': '第一个数字', 'title': 'first', 'type': 'integer'}, 'b': {'description': '第二个数字', 'title': 'second', 'type': 'integer'}}
print(f"custom_multiply.args = {custom_multiply.args}")

print("=============== 自定义描述：异步工具 ==================")


@tool
async def async_multiply(a: int, b: int) -> int:
    """这是一个用来进行两个数字相乘的工具"""
    return a * b


# async_multiply
print(f"async_multiply.name = {async_multiply.name}")
# 这是一个用来进行两个数字相乘的工具
print(f"async_multiply.description = {async_multiply.description}")
# {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
print(f"async_multiply.args = {async_multiply.args}")

# 调用方式参考：tools_structured_tool.py