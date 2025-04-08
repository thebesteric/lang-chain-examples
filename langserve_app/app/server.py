import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langserve import add_routes
from starlette.middleware.cors import CORSMiddleware

os.environ['LANGSMITH_PROJECT'] = 'langserve_app'

app = FastAPI(
    title="LangServe 服务器",
    description="使用 Langchain 的 Runnable 接口的简单 API 服务器",
    version="1.0.0",
)

add_routes(
    app,
    ChatOllama(model='llama3.1:8b', temperature=0.0),
    path="/chat",
)

add_routes(
    app,
    ChatOllama(model='llama3.1:8b', temperature=0.0) | StrOutputParser(),
    path="/chat_str_parser",
)

prompt = ChatPromptTemplate.from_template("告诉我一个关于{topic}的笑话")
add_routes(
    app,
    prompt | ChatOllama(model='llama3.1:8b', temperature=0.0),
    path="/chat_with_prompt",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Restful API
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
