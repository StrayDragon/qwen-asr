# Qwen ASR OpenAI-Compatible API Server

OpenAI Audio API 兼容的 Qwen ASR 语音转文字服务器。

## 功能特性

- ✅ OpenAI API 兼容 (`/v1/audio/transcriptions`)
- ✅ 支持流式和非流式响应
- ✅ Bearer token 认证
- ✅ 多模型支持 (0.6B / 1.7B)
- ✅ 模型池（并发请求）
- ✅ SSE (Server-Sent Events) 流式输出
- ✅ 懒加载与自动卸载（空闲超时释放内存）

## 快速开始

### 本地开发

```bash
# 1. 构建共享库
just libqwen

# 2. 安装 Python 依赖
just install-deps

# 3. 启动服务器
just serve
```

服务器将在 http://localhost:8011 启动。

API 文档: http://localhost:8011/docs

### Docker 部署

```bash
# 使用 docker-compose
docker compose up -d

# 或直接构建
docker build -t qwen-asr-openai-api .
docker run -p 8011:8011 -v ./qwen3-asr-0.6b:/app/qwen3-asr-0.6b:ro qwen-asr-openai-api
```

## API 使用

### 非流式请求

```bash
curl -X POST http://localhost:8011/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-test-key" \
  -F "file=@audio.wav" \
  -F "model=qwen-asr-0.6b"
```

响应：
```json
{
  "text": "transcribed text",
  "usage": {
    "type": "tokens",
    "input_tokens": 14,
    "output_tokens": 45,
    "total_tokens": 59
  }
}
```

### 流式请求 (SSE)

```bash
curl -X POST http://localhost:8011/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-test-key" \
  -F "file=@audio.wav" \
  -F "model=qwen-asr-0.6b" \
  -F "stream=true"
```

SSE 事件：
```
data: {"type": "transcript.text.delta", "delta": "text fragment"}

data: {"type": "transcript.text.done", "text": "full text", "usage": {...}}
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|---------|------|
| `QWEN_HOST` | 0.0.0.0 | 监听地址 |
| `QWEN_PORT` | 8011 | 监听端口 |
| `QWEN_MODEL_POOL_SIZE` | 2 | 每个模型的并发实例数 |
| `QWEN_MODEL_IDLE_TIMEOUT` | 300 | 模型空闲超时（秒），0=禁用自动卸载 |
| `QWEN_API_TOKEN` | sk-test-key | API 认证 token |

### 懒加载与自动卸载

服务器采用懒加载策略：
- 启动时不加载模型，仅在首次请求时加载
- 空闲超过 `QWEN_MODEL_IDLE_TIMEOUT` 秒后自动卸载模型释放内存
- 下次请求时自动重新加载

示例：
```bash
# 禁用自动卸载（模型常驻内存）
export QWEN_MODEL_IDLE_TIMEOUT=0

# 1 分钟无请求后卸载
export QWEN_MODEL_IDLE_TIMEOUT=60
```

## 请求参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|
| `file` | file | ✅ | 音频文件 (WAV, FLAC, MP3, M4A, OGG) |
| `model` | string | ✅ | 模型 ID (`qwen-asr-0.6b` 或 `qwen-asr-1.7b`) |
| `language` | string | ❌ | ISO-639-1 语言代码 (如 `zh`, `en`) |
| `prompt` | string | ❌ | 系统提示词 |
| `response_format` | string | ❌ | 响应格式，默认 `json` |
| `stream` | boolean | ❌ | 是否启用流式输出 |

## 测试

```bash
# 运行测试客户端
cd openai-compact-server
uv run python tests/test_client.py ../samples/jfk.wav

# 流式测试
uv run python tests/test_client.py --stream ../samples/jfk.wav

# 列出可用模型
uv run python tests/test_client.py --models
```

## 支持的音频格式

- 主要: WAV, FLAC (通过 soundfile，零拷贝)
- 次要: MP3, M4A, OGG (通过 librosa/ffmpeg)

目标格式：float32, 16kHz, 单声道

## 认证

默认 token: `sk-test-key`

生产环境请通过环境变量设置：
```bash
export QWEN_API_TOKEN="your-custom-token"
docker compose up -d
```

## 模型

客户端发送模型 ID，服务器映射到模型目录：

| 客户端请求 | 服务器路径 |
|------------|----------|
| `qwen-asr-0.6b` | `../qwen3-asr-0.6b/` |
| `qwen-asr-1.7b` | `../qwen3-asr-1.7b/` |

可扩展 `config.py` 中的 `MODELS` 字典添加自定义路径。
