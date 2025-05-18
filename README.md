# LLM-Pools

LLM-Pools是一个高性能的LLM请求路由与负载均衡系统，为大型语言模型API提供高效的负载分配、并发管理和队列处理。

## 更新说明 (2025-05-18)

### 并发处理优化
- 改进了队列管理器，支持更灵活地分配请求到不同模型池
- 修复了多个并发请求无法同时处理的问题
- 增强了负载计数的准确性和并发处理能力
- 增加了请求处理过程中的等待时间，确保负载计数充分更新

### 日志增强
- 增加了详细的负载状态日志记录
- 添加了实例状态变更的完整日志
- 增强了队列处理过程的日志详情
- 记录了请求全生命周期中的负载变化

### 池调度优化
- 实现了小模型池和大模型池的灵活调度
- 当一个池空闲时，可以处理另一个池的请求
- 优化了请求分配算法，优先使用负载较低的实例
- 增强了池状态监控和记录

## 功能特点

1. **多级模型池**: 支持大模型池和小模型池分离，可根据不同需求路由请求
2. **智能调度**: 根据实例负载和可用性自动选择最优LLM服务
3. **请求队列**: 当所有LLM实例都在高负载时，将请求放入队列等待处理
4. **优先级排序**: 支持基于用户级别的队列优先级
5. **并发控制**: 精确控制每个LLM实例的最大并发请求数
6. **健康检查**: 自动检测LLM服务的可用性并进行恢复
7. **请求重试**: 自动重试失败的请求，支持回退到备选模型
8. **详细日志**: 完整记录请求处理过程、队列状态和LLM使用情况

## 部署与配置

请参考`config.json`文件进行配置，可以设置不同模型类型的API端点、并发数和API密钥。

## 使用方法

1. 安装依赖: `pip install -r requirements.txt`
2. 配置API密钥和端点: 编辑`config.json`
3. 启动服务: `python main.py`

## API请求示例

```python
import requests
import json

# 向LLM-Pools发送请求
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "model": "large",  # 可选: "large", "small"
        "messages": [
            {"role": "user", "content": "请简要介绍量子计算的基本原理"}
        ],
        "stream": False,
        "user_level": 5  # 用户优先级(0-10)
    }
)

print(json.dumps(response.json(), ensure_ascii=False, indent=2))
```

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