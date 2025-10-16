from kg_gen.models import Graph  # noqa: F401
from kg_gen import KGGen
import json  # noqa: F401
import os  # noqa: F401

text = """
A Place for Demons
IT WAS FELLING NIGHT, and the usual crowd had gathered at the
Waystone Inn. Five wasn't much of a crowd, but five was as many as the
Waystone ever saw these days, times being what they were.
Old Cob was filling his role as storyteller and advice dispensary. The
men at the bar sipped their drinks and listened. In the back room a young
innkeeper stood out of sight behind the door, smiling as he listened to the
details of a familiar story.
"When he awoke, Taborlin the Great found himself locked in a high
tower. They had taken his sword and stripped him of his tools: key, coin,
and candle were all gone. But that weren't even the worst of it, you see…"
Cob paused for effect, "…cause the lamps on the wall were burning blue!"
Graham, Jake, and Shep nodded to themselves. The three friends had
grown up together, listening to Cob's stories and ignoring his advice.
Cob peered closely at the newer, more attentive member of his small
audience, the smith's prentice. "Do you know what that meant, boy?"
Everyone called the smith's prentice "boy" despite the fact that he was a
hand taller than anyone there. Small towns being what they are, he would
most likely remain "boy" until his beard filled out or he bloodied someone's
nose over the matter.
"""

# ========== 本地模型配置选项 ==========
# 根据你的需求选择以下配置之一：

# 选项1: 使用 Ollama 本地模型 (推荐)
# 需要先安装并启动 Ollama: https://ollama.ai/
# 然后下载模型: ollama pull llama3.2
LOCAL_MODEL_CONFIG = {
    "model": "ollama/llama3.2",  # 可选: llama3.1, qwen2.5:7b, mistral:7b 等
    "api_base": "http://localhost:11434",  # Ollama 默认端口
    "api_key": None,  # Ollama 不需要 API key
}

# 选项2: 使用 HuggingFace 模型 (需要较多内存)
# HF_MODEL_CONFIG = {
#     "model": "huggingface/microsoft/DialoGPT-medium",
#     "api_key": os.getenv("HF_TOKEN"),  # 可选，某些模型需要
# }

# 选项3: 使用 VLLM 本地部署 (高性能)
# 需要先安装 vllm: pip install vllm
# VLLM_MODEL_CONFIG = {
#     "model": "vllm/meta-llama/Llama-2-7b-chat-hf",
#     "api_base": "http://localhost:8000/v1",
# }

# 选项4: 使用原始 OpenAI GPT 模型 (需要 API key)
# OPENAI_MODEL_CONFIG = {
#     "model": "openai/gpt-4o",
#     "api_key": os.getenv("OPENAI_API_KEY"),
# }

# ========== 初始化 KGGen ==========
print("正在初始化 KGGen，使用本地模型...")
print(f"模型: {LOCAL_MODEL_CONFIG['model']}")

# 根据用户规则，如果需要从 HuggingFace 下载，设置镜像
if "huggingface" in LOCAL_MODEL_CONFIG.get("model", ""):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

kg = KGGen(
    model=LOCAL_MODEL_CONFIG["model"],
    api_key=LOCAL_MODEL_CONFIG.get("api_key"),
    api_base=LOCAL_MODEL_CONFIG.get("api_base"),
    temperature=0.0,
    max_tokens=4000,
)

# with open("tests/data/kingkiller_chapter_one.txt", "r", encoding="utf-8") as f:
#     text = f.read()

print("开始生成知识图谱...")
try:
    graph = kg.generate(
        input_data=text,
        chunk_size=1000,
        cluster=True,
        context="Kingkiller Chronicles",
        output_folder="./examples/",
    )
    
    print("知识图谱生成成功！")
    print(f"实体数量: {len(graph.entities)}")
    print(f"关系数量: {len(graph.relations)}")
    
    # 生成可视化
    print("正在生成可视化文件...")
    KGGen.visualize(graph, "./examples/basic-graph.html", True)
    print("可视化文件已保存到: ./examples/basic-graph.html")
    
except Exception as e:
    print(f"生成知识图谱时出错: {e}")
    print("\n故障排除提示:")
    print("1. 如果使用 Ollama，请确保已安装并运行: ollama serve")
    print("2. 如果使用 Ollama，请确保已下载模型: ollama pull llama3.2")
    print("3. 如果使用 HuggingFace 模型，请确保有足够的内存")
    print("4. 检查网络连接和防火墙设置")

# with open("./examples/graph.json", "r") as f:
#     graph = Graph(**json.load(f))