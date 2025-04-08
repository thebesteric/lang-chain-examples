import os
import time

from langchain.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model='qwen2.5:7b', temperature=0.0)

# 设置缓存到内存中
set_llm_cache(InMemoryCache())

def measure_invoke_time(llm, user_input):
    # 记录开始时间
    start_wall_time = time.time()
    start_cpu_times = os.times()

    # 调用模型
    chain = llm | StrOutputParser()
    response = chain.invoke(user_input)

    # 记录结束时间
    end_wall_time = time.time()
    end_cpu_times = os.times()

    # 计算时间差
    wall_time = end_wall_time - start_wall_time
    user_time = end_cpu_times.user - start_cpu_times.user
    sys_time = end_cpu_times.system - start_cpu_times.system
    total_cpu_time = user_time + sys_time

    return response, wall_time, user_time, sys_time, total_cpu_time


"""
必须问题一摸一样，才能命中缓存
"""


# 第一次调用
response1, wall_time1, user_time1, sys_time1, total_cpu_time1 = measure_invoke_time(llm, "给我讲一个笑话")
print(f"第一次调用结果：{response1}")
print(f"第一次调用，Wall Time: {wall_time1:.4f}秒，User Time: {user_time1:.4f}秒，System Time: {sys_time1:.4f}秒，Total CPU Time: {total_cpu_time1:.4f}秒")

# 第二次调用
response2, wall_time2, user_time2, sys_time2, total_cpu_time2 = measure_invoke_time(llm, "给我讲一个笑话")
print(f"第二次调用结果：{response2}")
print(f"第二次调用，Wall Time: {wall_time2:.4f}秒，User Time: {user_time2:.4f}秒，System Time: {sys_time2:.4f}秒，Total CPU Time: {total_cpu_time2:.4f}秒")

"""
第一次调用结果：当然可以！这是一个简单的笑话：
为什么电脑经常生病？
因为它的窗户（Windows）太多！
第一次调用，Wall Time: 2.9681秒，User Time: 0.0200秒，System Time: 0.0100秒，Total CPU Time: 0.0300秒

第二次调用结果：当然可以！这是一个简单的笑话：
为什么电脑经常生病？
因为它的窗户（Windows）太多！
第二次调用，Wall Time: 0.0015秒，User Time: 0.0000秒，System Time: 0.0000秒，Total CPU Time: 0.0000秒
"""