import json

import requests

print("同步调用 /chat_ext/invoke 结果")
response = requests.post(
    url="http://localhost:8000/chat_prompt/invoke",
    json={"input": {"topic": "猫"}}
)
print(response.json())


print("同步调用 /chat_prompt/stream 结果")
response = requests.post(
    url="http://localhost:8000/chat_with_prompt/stream",
    json={"input": {"topic": "猫"}}
)
for line in response.iter_lines():
    line = line.decode("utf-8")
    if line.startswith("data:") and not line.endswith("[DONE]"):
        data = json.loads(line[len("data: "):])
        print(data)
