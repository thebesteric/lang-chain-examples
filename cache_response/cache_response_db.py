import os
import time

from langchain.globals import set_llm_cache
from langchain.storage import redis
from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 必须使用 ChatOpenAI 来创建 llm
llm = ChatOpenAI(model='qwen2.5:7b', temperature=0.0, base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))

# 设置缓存到内存中
cache = SQLiteCache(database_path="./cache.db")

set_llm_cache(cache)

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