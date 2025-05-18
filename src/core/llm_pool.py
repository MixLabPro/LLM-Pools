import threading
import time
import uuid
import httpx
import json
import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Callable
from retry import retry
import random
from openai import AzureOpenAI, OpenAI

class LLMInstance:
    """
    LLM实例类，表示一个LLM API连接
    """
    
    def __init__(self, url: str, model: str, api_key: str, max_concurrency: int = 3):
        """
        初始化LLM实例
        
        Args:
            url: LLM API的URL端点
            model: 模型名称
            api_key: API密钥
            max_concurrency: 最大并发数
        """
        self.url = url
        self.model = model
        self.api_key = api_key
        self.max_concurrency = max_concurrency
        
        # 当前并发请求数
        self.current_concurrency = 0
        # 并发锁 (使用异步锁)
        self.concurrency_lock = asyncio.Lock()
        # 累计请求数
        self.total_requests = 0
        # 是否可用
        self.available = True
        # 最后一次失败时间
        self.last_failure_time = 0
        
        # 创建HTTP客户端
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        
        logging.info(f"已创建LLM实例: {model} @ {url}, 最大并发: {max_concurrency}")
    
    def __str__(self) -> str:
        """返回LLM实例的字符串表示（不包含API密钥）"""
        return f"{self.model} ({self.url})"
    
    async def increment_concurrency(self) -> bool:
        """
        增加并发计数，如果已达到最大并发数则返回False
        
        Returns:
            是否成功增加并发计数
        """
        async with self.concurrency_lock:
            # 记录增加前的状态
            old_concurrency = self.current_concurrency
            
            if self.current_concurrency >= self.max_concurrency or not self.available:
                logging.warning(f"【实例负载控制】LLM {self} 增加并发计数失败: 当前状态={old_concurrency}/{self.max_concurrency}, 可用状态={self.available}")
                return False
            
            self.current_concurrency += 1
            self.total_requests += 1
            
            # 记录增加后的状态
            logging.info(f"【实例负载控制】LLM {self} 并发数增加: {old_concurrency} -> {self.current_concurrency}/{self.max_concurrency}")
            return True
    
    async def decrement_concurrency(self) -> None:
        """
        减少并发计数
        """
        async with self.concurrency_lock:
            old_concurrency = self.current_concurrency
            
            if self.current_concurrency > 0:
                self.current_concurrency -= 1
                logging.info(f"【实例负载控制】LLM {self} 并发数减少: {old_concurrency} -> {self.current_concurrency}/{self.max_concurrency}")
            else:
                logging.warning(f"【实例负载控制】LLM {self} 并发计数已为零，无法减少")
    
    async def mark_unavailable(self) -> None:
        """
        标记LLM实例为不可用
        """
        async with self.concurrency_lock:
            self.available = False
            self.last_failure_time = time.time()
            logging.warning(f"LLM {self} 已标记为不可用")
    
    async def mark_available(self) -> None:
        """
        标记LLM实例为可用
        """
        async with self.concurrency_lock:
            self.available = True
            logging.info(f"LLM {self} 已恢复可用")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取LLM实例状态
        
        Returns:
            状态字典
        """
        # 由于get_status通常是同步调用，我们直接返回当前状态而不使用异步锁
        return {
            "model": self.model,
            "url": self.url,
            "current_concurrency": self.current_concurrency,
            "max_concurrency": self.max_concurrency,
            "total_requests": self.total_requests,
            "available": self.available
        }
    
    @retry(tries=2, delay=0.5, backoff=2)
    async def send_request(
        self, 
        endpoint: str, 
        payload: Dict[str, Any], 
        stream: bool = False
    ) -> Tuple[Any, Optional[str]]:
        """
        发送请求到LLM API
        
        Args:
            endpoint: API端点路径
            payload: 请求负载
            stream: 是否使用流式响应
            
        Returns:
            (响应数据, 错误消息)
        """
        try:
            # 确保模型名称正确
            if "model" in payload:
                payload["model"] = self.model
            
            # 移除不兼容参数
            clean_payload = payload.copy()
            incompatible_params = [
                "max_context_length",  # 常见API不支持
            ]
            for param in incompatible_params:
                if param in clean_payload:
                    logging.info(f"从请求中移除不兼容参数: {param}")
                    clean_payload.pop(param)
            
            # 构建完整URL
            url = f"{self.url}/{endpoint.lstrip('/')}"
            
            # 记录请求开始和请求内容
            logging.debug(f"发送请求到 {self}: {endpoint}, stream={stream}")
            
            # 记录完整请求内容，但敏感信息处理
            log_payload = clean_payload.copy()
            if "api_key" in log_payload:
                log_payload["api_key"] = "*****"  # 隐藏API密钥
            
            # 记录用户消息内容
            if "messages" in clean_payload:
                messages_log = "\n  ".join([
                    f"[{i+1}] {msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    for i, msg in enumerate(clean_payload["messages"])
                    if msg.get('content')
                ])
                if messages_log:
                    logging.info(f"用户消息内容:\n  {messages_log}")
            
            logging.debug(f"请求内容: {json.dumps(log_payload, ensure_ascii=False)}")
            
            start_time = time.time()
            
            # 发送请求
            if stream:
                # 使用send方法和stream=True参数进行流式请求
                request = self.client.build_request("POST", url, json=clean_payload)
                response = await self.client.send(request, stream=True)
                response.raise_for_status()
                
                # 读取并返回流式响应
                chunks = []
                combined_content = ""
                
                # 异步迭代流式响应
                async for chunk in response.aiter_lines():
                    if chunk.strip():
                        if chunk.startswith('data: '):
                            chunk = chunk[6:]  # 移除 'data: ' 前缀
                        
                        # 跳过[DONE]消息
                        if chunk == '[DONE]':
                            continue
                        
                        try:
                            # 解析JSON
                            chunk_data = json.loads(chunk)
                            chunks.append(chunk_data)
                            
                            # 提取并累积内容用于日志
                            if "choices" in chunk_data:
                                for choice in chunk_data["choices"]:
                                    if "delta" in choice and "content" in choice["delta"]:
                                        delta_content = choice["delta"].get("content", "")
                                        if delta_content:
                                            combined_content += delta_content
                        except json.JSONDecodeError as e:
                            logging.warning(f"无法解析流式响应块: {e}")
                
                # 记录请求相关信息
                elapsed_time = time.time() - start_time
                logging.info(f"流式请求完成, 耗时: {elapsed_time:.2f}秒, 收到 {len(chunks)} 个数据块")
                
                # 截取部分内容记录
                if combined_content:
                    max_log_length = 500  # 记录最大长度
                    log_content = combined_content[:max_log_length]
                    if len(combined_content) > max_log_length:
                        log_content += "..."
                    logging.info(f"响应内容: {log_content}")
                
                return chunks, None
            else:
                # 非流式请求
                response = await self.client.post(url, json=clean_payload)
                response.raise_for_status()
                
                # 解析JSON响应
                response_data = response.json()
                
                # 记录请求相关信息
                elapsed_time = time.time() - start_time
                logging.info(f"请求完成, 耗时: {elapsed_time:.2f}秒")
                
                # 记录生成的token数
                if "usage" in response_data:
                    usage = response_data["usage"]
                    logging.info(
                        f"Token使用: 输入={usage.get('prompt_tokens', 0)}, "
                        f"输出={usage.get('completion_tokens', 0)}, "
                        f"总计={usage.get('total_tokens', 0)}"
                    )
                
                # 记录部分响应内容
                if "choices" in response_data:
                    for choice in response_data["choices"]:
                        if "message" in choice and "content" in choice["message"]:
                            content = choice["message"].get("content", "")
                            if content:
                                max_log_length = 500  # 记录最大长度
                                log_content = content[:max_log_length]
                                if len(content) > max_log_length:
                                    log_content += "..."
                                logging.info(f"响应内容: {log_content}")
                
                return response_data, None
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP错误: {e.response.status_code} {e.response.reason_phrase}"
            try:
                error_json = e.response.json()
                if "error" in error_json:
                    error_msg += f" - {error_json['error'].get('message', 'Unknown error')}"
            except:
                error_msg += f" - 响应内容: {e.response.text}"
            
            logging.error(f"{self} 请求失败: {error_msg}")
            return None, error_msg
            
        except httpx.RequestError as e:
            error_msg = f"请求错误: {str(e)}"
            logging.error(f"{self} 请求失败: {error_msg}")
            return None, error_msg
            
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logging.error(f"{self} 请求失败: {error_msg}")
            return None, error_msg
        
        finally:
            # 无论成功失败，都减少并发计数
            await self.decrement_concurrency()
    
    async def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            是否健康
        """
        try:
            # 简单测试请求
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello, are you available?"}
            ]
            
            # 发送最小测试请求
            test_payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 5,
                "temperature": 0
            }
            
            # 发送健康检查请求
            response, error = await self.send_request("chat/completions", test_payload)
            
            if error:
                logging.warning(f"LLM {self} 健康检查失败: {error}")
                await self.mark_unavailable()
                return False
            
            # 如果成功，标记为可用
            await self.mark_available()
            return True
            
        except Exception as e:
            logging.warning(f"LLM {self} 健康检查失败: {str(e)}")
            await self.mark_unavailable()
            return False
    
    async def close(self) -> None:
        """
        关闭HTTP客户端
        """
        await self.client.aclose()
        logging.info(f"已关闭LLM实例: {self}")


class AzureLLMInstance(LLMInstance):
    """
    Azure OpenAI LLM实例类，专门处理Azure OpenAI API调用
    """
    
    def __init__(self, url: str, model: str, api_key: str, max_concurrency: int = 3):
        """
        初始化Azure LLM实例
        
        Args:
            url: Azure OpenAI URL端点
            model: 模型名称
            api_key: API密钥
            max_concurrency: 最大并发数
        """
        super().__init__(url, model, api_key, max_concurrency)
        
        # 解析Azure特定信息
        self._parse_azure_url()
        
        # 创建Azure OpenAI客户端替代HTTP客户端
        self.azure_client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=api_key,
            api_version=self.api_version
        )
        
        logging.info(f"已创建Azure LLM实例: {model} @ {self.azure_endpoint}, 部署ID: {self.deployment_id}, 最大并发: {max_concurrency}")
    
    def _parse_azure_url(self):
        """
        从URL中解析Azure特定信息：端点、部署ID和API版本
        """
        # 使用正则表达式提取端点和部署名称
        # 格式: https://xxxxx.openai.azure.com/openai/deployments/yyyy/chat/completions?api-version=zzzz
        pattern = r'https://([^/]+)/openai/deployments/([^/]+)/'
        match = re.match(pattern, self.url)
        
        if not match:
            raise ValueError(f"无法从URL解析出Azure端点和部署名: {self.url}")
        
        self.azure_endpoint = f"https://{match.group(1)}"
        self.deployment_id = match.group(2)
        
        # 尝试从URL中提取API版本
        self.api_version = "2024-10-21"  # 默认API版本
        api_version_match = re.search(r'api-version=([^&]+)', self.url)
        if api_version_match:
            self.api_version = api_version_match.group(1)
    
    @retry(tries=2, delay=0.5, backoff=2)
    async def send_request(
        self, 
        endpoint: str, 
        payload: Dict[str, Any], 
        stream: bool = False
    ) -> Tuple[Any, Optional[str]]:
        """
        使用Azure OpenAI客户端发送请求
        
        Args:
            endpoint: API端点路径（对于Azure OpenAI，这个参数会被忽略）
            payload: 请求负载
            stream: 是否使用流式响应
            
        Returns:
            (响应数据, 错误消息)
        """
        try:
            # 增加并发计数
            async with self.concurrency_lock:
                self.current_concurrency += 1
                self.total_requests += 1
                logging.debug(f"Azure LLM {self} 并发数增加至 {self.current_concurrency}/{self.max_concurrency}")
            
            # 记录请求开始
            logging.debug(f"发送请求到 Azure {self}: {self.deployment_id}, stream={stream}")
            
            # 记录用户消息内容
            if "messages" in payload:
                messages_log = "\n  ".join([
                    f"[{i+1}] {msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    for i, msg in enumerate(payload["messages"])
                    if msg.get('content')
                ])
                if messages_log:
                    logging.info(f"用户消息内容:\n  {messages_log}")
            
            start_time = time.time()
            
            # 发送Azure OpenAI请求
            try:
                # 移除不兼容参数
                clean_payload = payload.copy()
                incompatible_params = [
                    "max_context_length",  # Azure OpenAI不支持
                ]
                for param in incompatible_params:
                    if param in clean_payload:
                        logging.info(f"从Azure请求中移除不兼容参数: {param}")
                        clean_payload.pop(param)
                
                if stream:
                    # 流式请求
                    # 确保移除stream参数
                    if "stream" in clean_payload:
                        del clean_payload["stream"]
                        
                    response = await asyncio.to_thread(
                        self.azure_client.chat.completions.create,
                        model=self.deployment_id,
                        messages=clean_payload["messages"],
                        stream=True,  # 显式设置stream参数
                        **{k: v for k, v in clean_payload.items() if k not in ["model", "messages"]}
                    )
                    
                    # 处理流式响应
                    chunks = []
                    combined_content = ""
                    
                    # 将同步迭代器转换为异步迭代
                    for chunk in response:
                        chunks.append(chunk)
                        
                        # 提取并累积内容用于日志
                        if hasattr(chunk, 'choices') and chunk.choices:
                            for choice in chunk.choices:
                                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                                    combined_content += choice.delta.content
                    
                    # 计算总处理时间
                    processing_time = time.time() - start_time
                    logging.debug(f"完成Azure流式请求，耗时: {processing_time:.2f}秒, 收到 {len(chunks)} 个块")
                    
                    # 记录累积的内容
                    if combined_content:
                        logging.info(f"Azure流式响应累积内容: {combined_content}")
                    
                    # 转换为与普通API兼容的格式
                    return chunks, None
                else:
                    # 非流式请求
                    # 确保移除stream参数
                    if "stream" in clean_payload:
                        del clean_payload["stream"]
                        
                    response = await asyncio.to_thread(
                        self.azure_client.chat.completions.create,
                        model=self.deployment_id,
                        messages=clean_payload["messages"],
                        stream=False,  # 显式设置stream=False
                        **{k: v for k, v in clean_payload.items() if k not in ["model", "messages"]}
                    )
                    
                    # 计算总处理时间
                    processing_time = time.time() - start_time
                    logging.debug(f"完成Azure非流式请求，耗时: {processing_time:.2f}秒")
                    
                    # 记录响应内容
                    if hasattr(response, 'choices') and response.choices:
                        content = response.choices[0].message.content
                        logging.info(f"Azure LLM响应内容: {content}")
                    
                    # 返回响应对象
                    return response, None
            
            except Exception as e:
                error_msg = f"Azure OpenAI API调用失败: {str(e)}"
                logging.error(f"Azure LLM {self} 请求失败: {error_msg}")
                return None, error_msg
        
        except Exception as e:
            error_msg = f"Azure发送请求异常: {str(e)}"
            logging.error(f"Azure LLM {self} 请求失败: {error_msg}")
            return None, error_msg
        
        finally:
            # 无论成功失败，都减少并发计数
            await self.decrement_concurrency()
    
    async def health_check(self) -> bool:
        """
        执行Azure LLM健康检查
        
        Returns:
            是否健康
        """
        try:
            # 构造简单请求
            messages = [{"role": "user", "content": "Hi"}]
            
            # 使用Azure OpenAI客户端发送请求
            response = await asyncio.to_thread(
                self.azure_client.chat.completions.create,
                model=self.deployment_id,
                messages=messages,
                max_tokens=5,
                timeout=5.0
            )
            
            # 如果成功，标记为可用
            await self.mark_available()
            return True
            
        except Exception as e:
            logging.warning(f"Azure LLM {self} 健康检查失败: {str(e)}")
            await self.mark_unavailable()
            return False
    
    async def close(self) -> None:
        """
        关闭Azure OpenAI客户端
        """
        if hasattr(self, 'client') and self.client:
            await self.client.aclose()
        logging.info(f"已关闭Azure LLM实例: {self}")


class LLMPool:
    """
    LLM连接池，管理多个LLM实例
    """
    
    def __init__(self, pool_type: str, max_concurrency_per_llm: int = 3):
        """
        初始化LLM连接池
        
        Args:
            pool_type: 池类型 ("large" 或 "small")
            max_concurrency_per_llm: 每个LLM实例的最大并发数
        """
        self.pool_type = pool_type
        self.max_concurrency_per_llm = max_concurrency_per_llm
        self.llm_instances: List[LLMInstance] = []
        
        # 定期健康检查的任务
        self.health_check_task = None
        self.running = True
        
        logging.info(f"已创建LLM连接池: {pool_type}")
    
    def add_llm_instance(self, url: str, model: str, api_key: str, max_concurrency: int) -> None:
        """
        添加LLM实例到连接池
        
        Args:
            url: LLM API的URL端点
            model: 模型名称
            api_key: API密钥
            max_concurrency: 最大并发数
        """
        # 根据URL类型选择实例类型
        if "azure.com" in url:
            instance = AzureLLMInstance(
                url=url,
                model=model,
                api_key=api_key,
                max_concurrency=max_concurrency
            )
        else:
            instance = LLMInstance(
                url=url,
                model=model,
                api_key=api_key,
                max_concurrency=max_concurrency
            )
            
        self.llm_instances.append(instance)
        logging.info(f"已添加LLM实例到{self.pool_type}池: {model} @ {url}")
    
    async def get_available_instance(self) -> Optional[LLMInstance]:
        """
        获取可用的LLM实例
        
        Returns:
            LLM实例，如果没有可用实例则返回None
        """
        # 记录获取前的池状态
        current_load = self.get_current_load()
        total_capacity = self.get_total_capacity()
        logging.info(f"{self.pool_type}池 - 获取实例前状态: {current_load}/{total_capacity}")
        
        # 创建可用实例列表和日志详情
        available_instances = []
        instance_details = []
        
        for i, instance in enumerate(self.llm_instances):
            status = "可用" if instance.available else "不可用"
            capacity = f"{instance.current_concurrency}/{instance.max_concurrency}"
            instance_details.append(f"  [{i+1}] {instance.model}: {capacity} ({status})")
            
            if instance.available and instance.current_concurrency < instance.max_concurrency:
                available_instances.append(instance)
        
        # 记录每个实例的详细信息
        logging.info(f"{self.pool_type}池 - 实例状态:\n" + "\n".join(instance_details))
        
        if not available_instances:
            logging.info(f"{self.pool_type}池 - 无可用实例")
            return None
        
        # 按当前并发数排序，优先使用负载较轻的实例
        sorted_instances = sorted(available_instances, key=lambda x: x.current_concurrency)
        
        # 尝试增加并发计数
        for instance in sorted_instances:
            if await instance.increment_concurrency():
                # 记录获取后的池状态
                new_load = self.get_current_load()
                logging.info(f"{self.pool_type}池 - 获取实例后状态: {new_load}/{total_capacity}, 使用实例: {instance.model}")
                return instance
        
        # 如果所有实例都无法增加并发计数
        logging.warning(f"{self.pool_type}池 - 所有实例均无法增加并发计数")
        return None
    
    def get_status(self) -> List[Dict[str, Any]]:
        """
        获取所有LLM实例的状态
        
        Returns:
            状态字典列表
        """
        return [instance.get_status() for instance in self.llm_instances]
    
    def get_total_capacity(self) -> int:
        """
        获取连接池总容量（最大并发请求数）
        
        Returns:
            最大并发请求数
        """
        return sum(instance.max_concurrency for instance in self.llm_instances)
    
    def get_current_load(self) -> int:
        """
        获取当前负载（当前并发请求数）
        
        Returns:
            当前并发请求数
        """
        return sum(instance.current_concurrency for instance in self.llm_instances)
    
    def is_full(self) -> bool:
        """
        检查连接池是否已满
        
        Returns:
            连接池是否已满
        """
        return self.get_current_load() >= self.get_total_capacity()
    
    async def start_health_checks(self, interval_seconds: int = 60) -> None:
        """
        启动定期健康检查
        
        Args:
            interval_seconds: 健康检查间隔时间(秒)
        """
        self.running = True
        
        async def health_check_task():
            while self.running:
                try:
                    logging.debug(f"执行{self.pool_type}池健康检查")
                    
                    # 检查所有不可用的实例
                    unavailable_instances = [
                        inst for inst in self.llm_instances if not inst.available
                    ]
                    
                    for instance in unavailable_instances:
                        # 如果上次失败时间超过5分钟，尝试恢复
                        if time.time() - instance.last_failure_time > 300:
                            await instance.health_check()
                    
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    logging.error(f"健康检查任务异常: {str(e)}")
                    await asyncio.sleep(interval_seconds)
        
        self.health_check_task = asyncio.create_task(health_check_task())
        logging.info(f"{self.pool_type}池健康检查任务已启动，间隔: {interval_seconds}秒")
    
    async def stop_health_checks(self) -> None:
        """
        停止健康检查任务
        """
        self.running = False
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
            logging.info(f"{self.pool_type}池健康检查任务已停止")
    
    async def close(self) -> None:
        """
        关闭连接池中的所有LLM实例
        """
        await self.stop_health_checks()
        for instance in self.llm_instances:
            await instance.close()
        self.llm_instances.clear()
        logging.info(f"{self.pool_type}池已关闭")


class LLMPoolManager:
    """
    LLM池管理器，管理大模型和小模型连接池
    """
    
    def __init__(self, max_concurrency_per_llm: int = 3):
        """
        初始化LLM池管理器
        
        Args:
            max_concurrency_per_llm: 每个LLM实例的最大并发数
        """
        self.large_pool = LLMPool("large", max_concurrency_per_llm)
        self.small_pool = LLMPool("small", max_concurrency_per_llm)
        
        # 保存最大并发数参数
        self.max_concurrency_per_llm = max_concurrency_per_llm
        
        # 重试设置
        self.max_retries = 3
        self.retry_delay = 0.1  # 秒
        
        logging.info(f"LLM池管理器已初始化，每实例最大并发: {max_concurrency_per_llm}")
    
    def add_models_from_config(self, large_models: List[Dict[str, str]], small_models: List[Dict[str, str]]) -> None:
        """
        从配置添加模型
        
        Args:
            large_models: 大模型配置列表
            small_models: 小模型配置列表
        """
        # 添加大模型
        for model_config in large_models:
            # 获取request参数作为最大并发数，如果不存在则使用默认值
            max_concurrency = int(model_config.get("request", self.max_concurrency_per_llm))
            
            self.large_pool.add_llm_instance(
                url=model_config["url"],
                model=model_config["model"],
                api_key=model_config["api_key"],
                max_concurrency=max_concurrency
            )
        
        # 添加小模型
        for model_config in small_models:
            # 获取request参数作为最大并发数，如果不存在则使用默认值
            max_concurrency = int(model_config.get("request", self.max_concurrency_per_llm))
            
            self.small_pool.add_llm_instance(
                url=model_config["url"],
                model=model_config["model"],
                api_key=model_config["api_key"],
                max_concurrency=max_concurrency
            )
        
        logging.info(f"已添加 {len(large_models)} 个大模型和 {len(small_models)} 个小模型")
    
    def set_retry_settings(self, max_retries: int, retry_delay_ms: int) -> None:
        """
        设置重试参数
        
        Args:
            max_retries: 最大重试次数
            retry_delay_ms: 重试延迟(毫秒)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay_ms / 1000.0  # 转换为秒
        logging.info(f"已设置重试参数: 最大重试 {max_retries} 次, 延迟 {retry_delay_ms} 毫秒")
    
    def get_pool(self, pool_type: str) -> LLMPool:
        """
        获取指定类型的模型池
        
        Args:
            pool_type: 池类型 ("large", "small" 或 "default")
            
        Returns:
            LLM连接池
        """
        if pool_type == "large" or pool_type == "default":
            return self.large_pool
        elif pool_type == "small":
            return self.small_pool
        else:
            # 对于非标准值，返回默认池(大模型池)
            logging.warning(f"未知的池类型: {pool_type}，使用默认大模型池")
            return self.large_pool
    
    async def start_health_checks(self, interval_seconds: int = 60) -> None:
        """
        启动所有池的健康检查
        
        Args:
            interval_seconds: 健康检查间隔时间(秒)
        """
        await self.large_pool.start_health_checks(interval_seconds)
        await self.small_pool.start_health_checks(interval_seconds)
        logging.info(f"所有池的健康检查已启动，间隔: {interval_seconds}秒")
    
    async def send_request_with_retry(
        self,
        request_id: str,
        pool_type: str,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False,
        logger: Any = None
    ) -> Tuple[Any, Optional[str], Optional[str]]:
        """
        发送请求到LLM，支持自动重试
        
        Args:
            request_id: 请求ID
            pool_type: 池类型 ("large", "small" 或 "default")
            endpoint: API端点路径
            payload: 请求负载
            stream: 是否使用流式响应
            logger: 日志记录器
            
        Returns:
            (响应数据, 错误消息, 使用的LLM名称)
        """
        # 清理请求负载，移除不兼容的参数
        payload = self._clean_payload(payload)
        
        # 获取目标池
        pool = self.get_pool(pool_type)
        
        # 尝试的LLM实例记录
        tried_instances = set()
        
        # 记录池状态
        large_pool_status = self.large_pool.get_status()
        small_pool_status = self.small_pool.get_status()
        large_capacity = self.large_pool.get_total_capacity()
        small_capacity = self.small_pool.get_total_capacity()
        large_load = self.large_pool.get_current_load()
        small_load = self.small_pool.get_current_load()
        
        logging.info(f"【负载状态】请求 {request_id} 发送前池状态 - 大模型: {large_load}/{large_capacity}, 小模型: {small_load}/{small_capacity}")
        
        # 记录实例详细状态
        large_instances_detail = []
        for i, inst in enumerate(large_pool_status):
            large_instances_detail.append(f"  [{i+1}] {inst['model']} @ {inst['url']}: {inst['current_concurrency']}/{inst['max_concurrency']} ({'可用' if inst['available'] else '不可用'})")
        
        small_instances_detail = []
        for i, inst in enumerate(small_pool_status):
            small_instances_detail.append(f"  [{i+1}] {inst['model']} @ {inst['url']}: {inst['current_concurrency']}/{inst['max_concurrency']} ({'可用' if inst['available'] else '不可用'})")
        
        if large_instances_detail:
            logging.info(f"【实例详情】请求 {request_id} - 大模型实例状态:\n" + "\n".join(large_instances_detail))
        if small_instances_detail:
            logging.info(f"【实例详情】请求 {request_id} - 小模型实例状态:\n" + "\n".join(small_instances_detail))
        
        # 获取当前池中所有可用实例
        all_pool_instances = [inst for inst in pool.llm_instances if inst.available]
        
        # 获取备选池中所有可用实例
        fallback_pool = self.small_pool if pool_type == "large" else self.large_pool
        all_fallback_instances = [inst for inst in fallback_pool.llm_instances if inst.available]
        
        # 合并所有可用实例
        all_available_instances = all_pool_instances + all_fallback_instances
        
        # 如果没有可用实例，直接返回错误
        if not all_available_instances:
            error_msg = f"所有LLM实例都不可用，请求失败"
            logging.error(f"【请求失败】{request_id}: {error_msg}")
            return None, error_msg, None
        
        # 按负载排序所有实例 - 优先选择负载较低的实例
        all_available_instances.sort(key=lambda x: x.current_concurrency)
        
        # 记录可用实例详情
        logging.info(f"【调度决策】请求 {request_id} - 可用LLM实例数: {len(all_available_instances)}")
        available_instances_log = []
        for i, inst in enumerate(all_available_instances):
            instance_type = "大模型" if inst in all_pool_instances else "小模型"
            available_instances_log.append(f"  [{i+1}] {inst.model} @ {inst.url}: {inst.current_concurrency}/{inst.max_concurrency} ({instance_type})")
        
        if available_instances_log:
            logging.info(f"【调度详情】请求 {request_id} - 当前可用实例状态（按负载排序）:\n" + "\n".join(available_instances_log))
        
        # 重试循环
        retry_count = 0
        while retry_count < self.max_retries and tried_instances != set(f"{inst.url}:{inst.model}" for inst in all_available_instances):
            try:
                # 选择未尝试过的实例中负载最小的
                instance = None
                for inst in all_available_instances:
                    instance_key = f"{inst.url}:{inst.model}"
                    if instance_key not in tried_instances:
                        instance = inst
                        break
                
                # 如果所有实例都已尝试过但未达到最大重试次数，重置并重试
                if not instance:
                    if retry_count < self.max_retries - 1:
                        # 所有实例都尝试过一次，但还有重试次数，从头开始
                        retry_count += 1
                        await asyncio.sleep(self.retry_delay * (2 ** retry_count))  # 指数退避
                        continue
                    else:
                        error_msg = f"所有LLM实例都已尝试，请求失败"
                        logging.error(f"【请求失败】{request_id}: {error_msg}")
                        return None, error_msg, None
                
                # 记录已尝试实例
                instance_key = f"{instance.url}:{instance.model}"
                tried_instances.add(instance_key)
                
                # 记录切换池的信息（如果适用）
                instance_type = "大模型" if instance in all_pool_instances else "小模型"
                if (pool_type == "large" and instance in all_fallback_instances) or \
                   (pool_type == "small" and instance in all_pool_instances):
                    fallback_msg = f"切换到{'小' if pool_type == 'large' else '大'}模型池"
                    logging.info(f"【池切换】请求 {request_id}: {fallback_msg}")
                    if logger:
                        logger.info(fallback_msg, request_id)
                
                # 尝试增加并发计数
                incremented = await instance.increment_concurrency()
                if not incremented:
                    logging.warning(f"【并发控制】请求 {request_id}: 无法增加实例 {instance.model} 的并发计数，当前并发: {instance.current_concurrency}/{instance.max_concurrency}")
                    continue
                
                # 记录使用的LLM
                llm_name = f"{instance.model} ({instance.url})"
                logging.info(f"【请求分配】请求 {request_id} 分配给 {instance_type} 实例: {llm_name}, 当前并发: {instance.current_concurrency}/{instance.max_concurrency}")
                
                if logger:
                    logger.info(f"使用LLM: {llm_name}", request_id)
                
                # 记录增加并发后的池状态
                new_large_load = self.large_pool.get_current_load()
                new_small_load = self.small_pool.get_current_load()
                logging.info(f"【负载变化】请求 {request_id} 分配后池状态 - 大模型: {new_large_load}/{large_capacity}, 小模型: {new_small_load}/{small_capacity}")
                
                # 发送请求
                start_time = time.time()
                response, error = await instance.send_request(endpoint, payload, stream)
                request_time = time.time() - start_time
                
                # 请求结束后，减少并发计数
                await instance.decrement_concurrency()
                
                # 记录减少并发后的池状态
                after_large_load = self.large_pool.get_current_load()
                after_small_load = self.small_pool.get_current_load()
                logging.info(f"【负载变化】请求 {request_id} 完成后池状态 - 大模型: {after_large_load}/{large_capacity}, 小模型: {after_small_load}/{small_capacity}, 请求耗时: {request_time:.2f}秒")
                
                if error:
                    # 如果失败，记录错误并继续重试
                    logging.error(f"【请求失败】请求 {request_id} LLM调用失败: {error}")
                    if logger:
                        logger.log_error_retry(
                            request_id=request_id,
                            llm_name=llm_name,
                            error_message=error,
                            retry_attempt=retry_count + 1,
                            max_retries=self.max_retries
                        )
                    
                    # 标记为不可用（如果是连接错误）
                    if "connection" in error.lower() or "timeout" in error.lower():
                        await instance.mark_unavailable()
                    
                    # 等待一段时间后继续
                    retry_count += 1
                    if retry_count < self.max_retries:
                        await asyncio.sleep(self.retry_delay * (2 ** retry_count))  # 指数退避
                    continue
                
                # 请求成功
                logging.info(f"【请求成功】请求 {request_id} 在 {instance_type} 实例 {instance.model} 上执行成功")
                return response, None, llm_name
            except Exception as e:
                error_msg = f"发送请求异常: {str(e)}"
                logging.error(f"【请求异常】{request_id}: {error_msg}")
                
                # 如果实例存在，减少并发计数
                if instance:
                    await instance.decrement_concurrency()
                    
                    # 记录减少并发后的池状态
                    after_large_load = self.large_pool.get_current_load()
                    after_small_load = self.small_pool.get_current_load()
                    logging.info(f"【负载变化】请求 {request_id} 异常后池状态 - 大模型: {after_large_load}/{large_capacity}, 小模型: {after_small_load}/{small_capacity}")
                
                # 记录重试信息
                if logger:
                    logger.log_error_retry(
                        request_id=request_id,
                        llm_name=f"{instance.model if instance else 'unknown'} ({instance.url if instance else 'unknown'})",
                        error_message=error_msg,
                        retry_attempt=retry_count + 1,
                        max_retries=self.max_retries
                    )
                
                # 等待一段时间后继续
                retry_count += 1
                if retry_count < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** retry_count))  # 指数退避
        
        # 所有重试都失败
        error_msg = f"达到最大重试次数 ({self.max_retries})，请求失败"
        logging.error(f"【请求失败】{request_id}: {error_msg}")
        return None, error_msg, None
    
    def _clean_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理请求负载，移除不兼容的参数
        
        Args:
            payload: 原始请求负载
            
        Returns:
            清理后的请求负载
        """
        # 创建请求负载的副本，避免修改原始对象
        clean_payload = payload.copy()
        
        # 移除已知的不兼容参数
        incompatible_params = [
            "max_context_length",  # Mistral和Azure OpenAI不支持
        ]
        
        for param in incompatible_params:
            if param in clean_payload:
                logging.info(f"从请求中移除不兼容参数: {param}")
                clean_payload.pop(param)
        
        return clean_payload
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取所有池的状态
        
        Returns:
            状态字典
        """
        # 获取基础状态
        large_pool_stats = {
            "instances": self.large_pool.get_status(),
            "current_load": self.large_pool.get_current_load(),
            "total_capacity": self.large_pool.get_total_capacity(),
            "is_full": self.large_pool.is_full(),
            "utilization_percent": round(self.large_pool.get_current_load() / max(1, self.large_pool.get_total_capacity()) * 100, 2),
            "instance_count": len(self.large_pool.llm_instances),
            "available_instance_count": sum(1 for inst in self.large_pool.llm_instances if inst.available)
        }
        
        small_pool_stats = {
            "instances": self.small_pool.get_status(),
            "current_load": self.small_pool.get_current_load(),
            "total_capacity": self.small_pool.get_total_capacity(),
            "is_full": self.small_pool.is_full(),
            "utilization_percent": round(self.small_pool.get_current_load() / max(1, self.small_pool.get_total_capacity()) * 100, 2),
            "instance_count": len(self.small_pool.llm_instances),
            "available_instance_count": sum(1 for inst in self.small_pool.llm_instances if inst.available)
        }
        
        # 统计总体情况
        total_stats = {
            "total_instances": large_pool_stats["instance_count"] + small_pool_stats["instance_count"],
            "available_instances": large_pool_stats["available_instance_count"] + small_pool_stats["available_instance_count"],
            "total_capacity": large_pool_stats["total_capacity"] + small_pool_stats["total_capacity"],
            "current_load": large_pool_stats["current_load"] + small_pool_stats["current_load"],
            "overall_utilization_percent": round(
                (large_pool_stats["current_load"] + small_pool_stats["current_load"]) / 
                max(1, (large_pool_stats["total_capacity"] + small_pool_stats["total_capacity"])) * 100, 
                2
            )
        }
        
        return {
            "large_pool": large_pool_stats,
            "small_pool": small_pool_stats,
            "overall": total_stats,
            "timestamp": time.time()
        }
    
    async def close(self) -> None:
        """
        关闭所有连接池
        """
        await self.large_pool.close()
        await self.small_pool.close()
        logging.info("LLM池管理器已关闭") 