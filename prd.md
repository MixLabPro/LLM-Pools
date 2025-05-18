# LLM路由池管理系统需求文档 (最终版)

## 1. 项目概述

开发一个类似routeLLM的服务，实现对多个LLM（大型语言模型）的统一管理和智能路由。该服务器将接收客户端的API请求，根据负载情况路由到合适的LLM服务，并将响应返回给客户端。系统将通过JSON配置管理所有LLM资源，并提供完全兼容OpenAI API的接口。

## 2. 功能需求

### 2.1 LLM配置管理

- **配置格式**：使用JSON格式配置所有LLM资源
- **配置内容**：每个LLM需包含以下信息：
  - URL端点
  - 模型名称/ID
  - API密钥
- **模型分类**：
  - 大模型组：高性能、高参数量的模型
  - 小模型组：轻量级、响应更快的模型
- **存储结构**：每种类型的模型以数组形式在JSON中存储

### 2.2 LLM池管理与路由功能

- **并发控制**：
  - 每个LLM实例同时允许3个并发请求
  - 当某个LLM达到并发上限时，自动路由至其他可用LLM
  
- **请求排队机制**：
  - 当所有LLM实例都达到并发上限时，新请求进入等待队列
  - 一旦有LLM实例并发数降低，队列中的请求按先进先出原则被处理
  - 排队请求需要设置超时时间，超时未处理则返回超时错误

- **故障处理与重试**：
  - 当LLM API调用失败时，自动切换到其他可用LLM实例重试
  - 每个请求最多尝试3次不同的LLM实例
  - 3次尝试全部失败后，向客户端返回明确的错误信息
  - 错误信息包含故障原因和已尝试的LLM信息（不包含API密钥）

- **负载均衡**：
  - 智能分配请求至未达到并发上限的模型
  - 在多个可用模型间实现请求的均衡分配
  - 优先选择当前并发数最低的LLM实例

- **资源最大化**：
  - 例如：系统配置了7个不同的LLM API密钥，理论上可支持最大21个并发请求

### 2.3 API接口设计

- **完全兼容OpenAI API**：
  - 实现与OpenAI API相同的端点和参数
  - 客户端可以直接使用标准OpenAI SDK连接此服务
  
- **扩展参数**：
  - 在`model`参数中支持特殊值：
    - `model="large"`：表示使用大模型池
    - `model="small"`：表示使用小模型池
    - `model="default"`或未指定：默认使用大模型池
  - `stream`参数：
    - 控制是否使用流式响应
    - 默认值为`false`（非流式响应）
    - 当设置为`true`时使用流式响应

### 2.4 日志系统

- **请求日志**：
  - 记录每个客户端请求的详细信息，包括：
    - 请求时间戳
    - 请求ID（唯一标识符）
    - 请求参数（model、stream等）
    - 请求内容摘要
  
- **LLM池状态日志**：
  - 在每次客户端请求到达时记录当前LLM池状态：
    - 每个LLM实例的当前并发数
    - 每个LLM实例的累计请求数
    - 当前等待队列长度
  
- **路由决策日志**：
  - 记录路由决策过程：
    - 选择的LLM实例
    - 选择原因
    - 如进入等待队列，记录等待位置和预计等待时间
  
- **错误与重试日志**：
  - 详细记录所有错误情况：
    - 失败的LLM调用详情（不包含API密钥）
    - 错误类型和错误消息
    - 重试次数和重试的LLM实例
    - 最终错误结果
  
- **性能指标日志**：
  - 记录系统性能相关指标：
    - 请求路由耗时
    - LLM响应时间
    - 总处理时间
    - 队列等待时间（如适用）

## 3. 系统架构

### 3.1 核心组件

- **HTTP服务器**：提供与OpenAI兼容的API端点
- **配置管理器**：负责读取和解析JSON配置
- **LLM连接池**：维护与各LLM的连接和并发状态
- **路由决策器**：根据当前负载和配置选择合适的LLM
- **请求转发器**：将请求转发给选定的LLM并处理响应
- **响应处理器**：处理流式和非流式响应的差异
- **日志管理器**：处理所有系统日志的记录和输出
- **队列管理器**：管理请求等待队列和超时控制

### 3.2 数据流

1. 客户端通过HTTP发送与OpenAI兼容的请求
2. 系统解析请求参数并决定使用大模型还是小模型池：
   - 如果未指定model参数或model="default"，使用大模型池
   - 如果model="large"，使用大模型池
   - 如果model="small"，使用小模型池
3. 系统确定响应类型：
   - 如果未指定stream参数或stream=false，使用非流式响应
   - 如果stream=true，使用流式响应
4. 日志系统记录请求信息和当前LLM池状态
5. 路由决策器检查所选池中的LLM并发情况：
   - 如果有可用LLM，选择并发数最低的LLM实例
   - 如果所有LLM都达到并发上限，将请求放入等待队列
6. 请求转发器将请求发送给选定的LLM
7. 如果LLM调用失败，系统自动重试其他LLM实例（最多3次）
8. 根据客户端请求的流式选项处理响应：
   - 如果是流式请求，实时转发LLM的流式响应给客户端
   - 如果是非流式请求，等待完整响应后一次性返回
9. 请求完成后，释放并发计数资源，并从等待队列中取出下一个请求处理
10. 日志系统记录完整的请求处理过程和结果

## 4. API端点设计

完全兼容OpenAI API的端点结构：

- **POST /v1/chat/completions**：聊天完成接口
- **POST /v1/completions**：文本完成接口
- **POST /v1/embeddings**：嵌入向量接口（如果支持）

## 5. 并发控制与请求排队详细设计

### 5.1 并发计数机制

- **计数器设计**：
  - 每个LLM实例维护一个原子计数器记录当前并发数
  - 计数器初始值为0，每接收一个请求加1，处理完成后减1
  - 计数器上限为3，达到上限后不再接收新请求

- **并发状态检查**：
  - 系统定期（如每秒）检查所有LLM实例的并发状态
  - 如发现某LLM实例长时间保持并发上限，可能触发健康检查

- **并发监控统计**：
  - 记录每个LLM实例的峰值并发数
  - 计算每个LLM实例的平均并发数
  - 统计并发饱和度（实际并发数/最大并发数）

### 5.2 请求排队系统

- **队列结构**：
  - 使用优先级队列存储等待请求
  - 默认按先进先出排序
  - 支持可选的优先级策略（如小型请求优先）

- **队列监控**：
  - 记录队列长度变化
  - 统计平均等待时间
  - 设置最大队列长度阈值，超出时拒绝新请求

- **超时机制**：
  - 每个入队请求设置最大等待时间（如30秒）
  - 超时未处理的请求自动从队列移除并返回超时错误
  - 支持客户端设置自定义超时时间

### 5.3 故障处理与重试策略

- **错误分类**：
  - 临时性错误（网络超时、服务暂时不可用）：自动重试
  - 永久性错误（认证失败、请求格式错误）：立即返回错误

- **重试机制**：
  - 最多尝试3次不同的LLM实例
  - 每次重试使用指数退避策略（如100ms, 200ms, 400ms）
  - 重试时优先选择不同供应商的LLM，避免单点故障

- **健康检查**：
  - 定期向LLM发送轻量级请求检查可用性
  - 连续失败的LLM实例暂时标记为不可用
  - 不可用LLM实例定期尝试恢复

- **降级策略**：
  - 当大模型池全部不可用时，可选择自动降级到小模型池
  - 所有模型不可用时，返回服务不可用错误

## 6. 配置示例

```json
{
  "large_models": [
    {
      "url": "https://api.example.com/v1",
      "model": "gpt-4",
      "api_key": "key1"
    },
    {
      "url": "https://api.another-provider.com/v1",
      "model": "claude-3",
      "api_key": "key2"
    }
  ],
  "small_models": [
    {
      "url": "https://api.example.com/v1",
      "model": "gpt-3.5-turbo",
      "api_key": "key3"
    },
    {
      "url": "https://api.llama.ai/v1",
      "model": "llama-7b",
      "api_key": "key4"
    }
  ],
  "queue_settings": {
    "max_queue_length": 100,
    "default_timeout": 30
  },
  "retry_settings": {
    "max_retries": 3,
    "retry_delay_ms": 100,
    "retry_multiplier": 2
  },
  "logging": {
    "level": "info",
    "file_path": "/var/log/llm-router/router.log",
    "rotate_size_mb": 10,
    "keep_logs_days": 7
  }
}
```

## 7. 客户端使用场景示例

客户端使用标准的OpenAI SDK，只需要将base_url指向路由服务器：

### 场景1：客户端初始化

```python
# 导入标准OpenAI SDK
from openai import OpenAI

# 初始化客户端，指向路由服务器
client = OpenAI(
    api_key="your-api-key",
    base_url="http://your-router-service.com/v1"  # 路由服务器地址
)
```

### 场景2：发送聊天请求（未指定model和stream）

```python
# 未指定model和stream参数，使用默认值
# model默认为default（大模型池）
# stream默认为false（非流式响应）
response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "帮我分析这个复杂问题。"}
    ],
    temperature=0.7
)

# 获取非流式响应
print(response.choices[0].message.content)
```

### 场景3：明确指定参数

```python
# 明确指定使用大模型池和流式响应
stream = client.chat.completions.create(
    model="large",  # 使用大模型池
    messages=[
        {"role": "user", "content": "请详细解释量子计算。"}
    ],
    stream=True  # 启用流式响应
)

# 处理流式响应
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### 场景4：使用小模型

```python
# 使用小模型池
response = client.chat.completions.create(
    model="small",  # 使用小模型池
    messages=[
        {"role": "user", "content": "简单计算2+2等于几？"}
    ]
    # 未指定stream，默认为false
)
```

## 8. 日志输出示例

### 请求到达日志
```
[2025-05-18 14:30:45.123] [INFO] [REQ-12345] New request received:
- Model: default (not specified, using large pool)
- Stream: false (not specified, using default)
- Content length: 256 bytes

[2025-05-18 14:30:45.124] [INFO] [REQ-12345] Current LLM pool status:
- Large pool:
  - gpt-4 (api.example.com): 2/3 concurrent requests
  - claude-3 (api.another-provider.com): 3/3 concurrent requests
- Small pool:
  - gpt-3.5-turbo (api.example.com): 1/3 concurrent requests
  - llama-7b (api.llama.ai): 0/3 concurrent requests
- Queue status: 0 requests waiting
```

### 路由决策日志
```
[2025-05-18 14:30:45.125] [INFO] [REQ-12345] Route decision:
- Selected LLM: gpt-4 (api.example.com)
- Current concurrency: 2/3
- Decision reason: lowest concurrency in large pool
```

### 错误和重试日志
```
[2025-05-18 14:30:46.500] [ERROR] [REQ-12345] LLM request failed:
- LLM: gpt-4 (api.example.com)
- Error: Connection timeout
- Retry attempt: 1/3

[2025-05-18 14:30:46.700] [INFO] [REQ-12345] Retrying with alternative LLM:
- New LLM: claude-3 (api.another-provider.com)
- Current status: Queue position 1 (waiting for capacity)
```

### 请求完成日志
```
[2025-05-18 14:30:50.800] [INFO] [REQ-12345] Request completed:
- LLM used: claude-3 (api.another-provider.com)
- Processing time: 4.2 seconds
- Tokens generated: 512
- Queue wait time: 1.3 seconds
```

## 9. 验收标准

- 服务器完全兼容OpenAI API，客户端可以使用标准OpenAI SDK与之交互
- 支持通过`model`参数选择大模型或小模型池（"large"、"small"或"default"）
- 当未指定`model`参数时，默认使用大模型池（等同于"default"）
- 当未指定`stream`参数时，默认使用非流式响应（等同于`stream=false`）
- 正确处理流式和非流式响应的差异
- 在高并发情况下正确分配请求到未达上限的LLM
- 当所有LLM达到并发上限时，新请求正确进入等待队列
- LLM调用失败时，自动重试最多3次不同的LLM实例
- 详细的日志系统记录每个请求的完整处理过程及LLM池状态
- 所有API响应格式与OpenAI保持兼容
- 错误处理和恢复机制正常工作

## 10. 未来扩展方向

- 增加更多OpenAI兼容的端点（如fine-tuning、images等）
- 实现更复杂的路由策略（基于内容、成本等）
- 提供管理界面监控模型使用情况
- 支持更多的LLM供应商和模型
- 增加自动伸缩能力，根据负载自动增减LLM实例
- 添加更复杂的队列优先级策略，如基于用户等级的优先级