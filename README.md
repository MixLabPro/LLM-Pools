# LLM-Pools

LLM-Pools是一个大语言模型API的连接池管理系统，用于高效地管理和分配多个LLM API实例之间的请求负载。

## 功能特点

- 管理多个LLM API连接
- 自动负载均衡
- 池中实例的健康检查
- 请求自动重试和故障转移
- 支持大模型和小模型两种池类型
- 随机选择算法，避免单点负载

## 项目结构

```
LLM-Pools/
├── src/
│   ├── core/
│   │   ├── llm_pool.py       # LLM池的核心实现
│   │   └── ...
│   ├── test_llm_pool_client.py # 测试客户端
│   └── ...
└── README.md
```

## 测试

### 安装依赖

```bash
pip install httpx asyncio tqdm prettytable retry
```

### 运行测试

测试客户端可以模拟多个并发请求，并监控LLM池的性能和负载分布：

```bash
python src/test_llm_pool_client.py
```

默认情况下，测试会向服务器发送20个并发请求，并实时显示请求进度和服务器状态。

### 测试配置

在`test_llm_pool_client.py`文件中可以修改以下配置：

- `SERVER_URL`: 服务器地址，默认为`http://localhost:8000`
- `API_KEY`: API密钥，默认为`test-api-key`
- `CONCURRENT_REQUESTS`: 并发请求数，默认为20
- `REQUEST_TIMEOUT`: 请求超时时间，默认为30秒

### 测试输出

测试会显示以下信息：

1. 服务器初始状态
2. 实时请求进度
3. 每个LLM实例的请求分布
4. 响应时间统计
5. 测试完成后的最终统计
6. 服务器最终状态 

# Azure OpenAI 客户端

这个项目提供了简单的Azure OpenAI API调用实现，从`config.json`自动读取配置信息。

## 文件说明

- `azure_client.py`: 核心客户端实现，提供Azure OpenAI客户端实例化和配置加载
- `azure_example.py`: 示例使用代码，展示如何调用Azure OpenAI API
- `config.json`: 配置文件，包含Azure端点和API密钥信息

## 使用方法

### 1. 直接运行示例

```bash
python azure_client.py  # 运行基本示例
python azure_example.py  # 运行详细示例
```

### 2. 在你的代码中使用

```python
from azure_client import get_azure_client

# 获取客户端和部署ID
client, deployment_id = get_azure_client()

# 调用API
response = client.chat.completions.create(
    model=deployment_id,  # 使用部署ID（不是模型名称）
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ]
)

# 获取响应内容
print(response.choices[0].message.content)
```

## 故障排除

如果遇到404错误 `Resource not found`，通常是因为以下原因：

1. Azure端点URL格式不正确
2. 部署ID不匹配 - 确保URL中的部署ID与实际部署名称一致
3. API版本不兼容 - 尝试修改`api_version`参数

## 配置文件格式

在`config.json`中配置Azure OpenAI服务的格式如下：

```json
{
  "large_models": [
    {
      "url": "https://你的资源名称.openai.azure.com/openai/deployments/部署名称/",
      "model": "部署名称",
      "api_key": "你的API密钥"
    }
  ]
}
```

注意URL必须包含`/openai/deployments/部署名称/`格式，客户端会自动解析出正确的端点和部署ID。 