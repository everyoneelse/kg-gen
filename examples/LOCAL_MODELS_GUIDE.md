# 本地模型使用指南

本指南将帮助你配置 KGGen 使用本地模型，而不是 OpenAI GPT 模型。

## 支持的本地模型类型

### 1. Ollama (推荐)

**优点**: 易于安装和使用，支持多种开源模型，资源占用适中
**缺点**: 需要额外安装 Ollama

#### 安装步骤:
```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. 启动 Ollama 服务
ollama serve

# 3. 下载模型 (选择其中一个)
ollama pull llama3.2        # 推荐，平衡性能和资源
ollama pull llama3.1:8b     # 更大的模型，更好的性能
ollama pull qwen2.5:7b      # 中文支持更好
ollama pull mistral:7b      # 轻量级选择
```

#### 配置代码:
```python
LOCAL_MODEL_CONFIG = {
    "model": "ollama/llama3.2",
    "api_base": "http://localhost:11434",
    "api_key": None,
}
```

### 2. HuggingFace Transformers

**优点**: 直接使用 transformers 库，无需额外服务
**缺点**: 需要较多内存，首次使用需要下载模型

#### 安装步骤:
```bash
pip install transformers torch
```

#### 配置代码:
```python
# 根据用户规则设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

HF_MODEL_CONFIG = {
    "model": "huggingface/microsoft/DialoGPT-medium",
    "api_key": os.getenv("HF_TOKEN"),  # 可选
}
```

### 3. VLLM (高性能)

**优点**: 高性能推理，支持批处理
**缺点**: 安装复杂，需要 GPU

#### 安装步骤:
```bash
pip install vllm
```

#### 配置代码:
```python
VLLM_MODEL_CONFIG = {
    "model": "vllm/meta-llama/Llama-2-7b-chat-hf",
    "api_base": "http://localhost:8000/v1",
}
```

## 修改后的 basic.py 使用方法

1. **选择模型配置**: 在 `basic.py` 中取消注释你想使用的模型配置
2. **运行脚本**: `python examples/basic.py`

## 常见问题排除

### Ollama 相关问题

**问题**: `Connection refused` 错误
**解决**: 确保 Ollama 服务正在运行: `ollama serve`

**问题**: 模型未找到
**解决**: 确保已下载模型: `ollama pull llama3.2`

**问题**: 端口冲突
**解决**: 修改 `api_base` 为其他端口，或停止占用 11434 端口的程序

### HuggingFace 相关问题

**问题**: 下载速度慢
**解决**: 已设置镜像 `HF_ENDPOINT=https://hf-mirror.com`

**问题**: 内存不足
**解决**: 选择更小的模型或增加系统内存

### 性能优化建议

1. **选择合适的模型大小**: 
   - 7B 模型: 需要约 14GB 内存
   - 13B 模型: 需要约 26GB 内存
   - 70B 模型: 需要约 140GB 内存

2. **调整参数**:
   - `chunk_size`: 较小的值可以减少内存使用
   - `temperature`: 设为 0.0 获得确定性结果
   - `max_tokens`: 根据需要调整输出长度

3. **硬件要求**:
   - CPU: 至少 4 核心
   - 内存: 至少 16GB (推荐 32GB)
   - 存储: 至少 20GB 可用空间

## 服务器环境配置

根据用户规则，如果在特定服务器上运行:

### 10.8.71.126 服务器
```bash
# 激活 Python 3.12 虚拟环境
source /path/to/py312/bin/activate
```

### 10.8.71.44 服务器
```bash
# 激活 Python 3.10 虚拟环境
source /path/to/py310/bin/activate
# 激活 C++ 开发工具
source /opt/rh/devtoolset-9/enable
```

## 模型推荐

| 用途 | 推荐模型 | 内存需求 | 性能 | max_tokens 限制 |
|------|----------|----------|------|----------------|
| 快速测试 | ollama/llama3.2 | 8GB | 中等 | 4000+ |
| 中文处理 | ollama/qwen2.5:7b | 14GB | 高 | 4000+ |
| 高质量输出 | ollama/llama3.1:8b | 16GB | 高 | 4000+ |
| 资源受限 | ollama/mistral:7b | 8GB | 中等 | 4000+ |
| 中文对话 | deepseek/deepseek-chat | 云端 | 高 | **8192** |

### 重要提示: max_tokens 限制

不同模型对 `max_tokens` 有不同的限制：

- **Deepseek 模型**: 最大 8192 tokens
- **GPT-4/GPT-3.5**: 通常 4000-8000 tokens  
- **开源模型 (Llama, Qwen 等)**: 通常 2000-4000 tokens
- **GPT-5 系列**: 最小 16000 tokens

如果遇到 `max_tokens` 相关错误，请：
1. 检查模型的 token 限制
2. 相应调整配置中的 `max_tokens` 值
3. 考虑使用 `chunk_size` 参数分块处理长文本

选择适合你的硬件配置和需求的模型即可开始使用！