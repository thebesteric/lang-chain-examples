from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 本地魔塔下载的模型路径
model_path = "/Users/wangweijun/AI/models/Qwen/Qwen2___5-0___5B-Instruct"

# 加载分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except Exception as e:
    print(f"加载分词器时出错: {e}")
    raise

# 加载模型，指定使用 safetensors
try:
    model = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=True)
except Exception as e:
    print(f"加载模型时出错: {e}")
    raise

# 创建文本生成管道
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7
)

# 使用 LangChain 的 HuggingFacePipeline 加载模型
llm = HuggingFacePipeline(pipeline=pipe)

# 输入文本进行推理
input_text = "你是谁"
try:
    result = llm.invoke(input_text)
    print(result)
except Exception as e:
    print(f"推理过程中出错: {e}")