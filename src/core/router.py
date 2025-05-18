import time
import uuid
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable

from src.core.llm_pool import LLMPoolManager
from src.core.queue_manager import QueueManager

class RouterManager:
    """
    路由决策器，根据当前负载和配置选择合适的LLM
    """
    
    def __init__(
        self,
        llm_pool_manager: LLMPoolManager,
        queue_manager: QueueManager,
        logger: Any = None
    ):
        """
        初始化路由决策器
        
        Args:
            llm_pool_manager: LLM池管理器
            queue_manager: 队列管理器
            logger: 日志管理器(可选)
        """
        self.llm_pool_manager = llm_pool_manager
        self.queue_manager = queue_manager
        self.logger = logger
        
        # 记录对应请求ID的处理时间
        self.request_start_times: Dict[str, float] = {}
        
        # 记录对应请求ID的处理结果回调
        self.request_callbacks: Dict[str, Callable] = {}
        
        # 处理队列的任务
        self.queue_processor_task = None
        self.running = True
        
        logging.info("路由决策器已初始化")
    
    def generate_request_id(self) -> str:
        """
        生成唯一请求ID
        
        Returns:
            请求ID
        """
        return f"req-{uuid.uuid4().hex[:8]}"
    
    async def route_request(
        self,
        request_data: Dict[str, Any],
        model: str = "default",
        stream: bool = False,
        timeout_seconds: Optional[int] = None,
        user_level: int = 0
    ) -> Tuple[Any, Optional[str], float]:
        """
        路由请求到合适的LLM
        
        Args:
            request_data: 请求数据
            model: 模型池类型 ("large", "small" 或 "default")
            stream: 是否使用流式响应
            timeout_seconds: 超时时间(秒)
            user_level: 用户等级(0-10)
            
        Returns:
            (响应数据, 错误消息, 处理时间)
        """
        # 生成请求ID
        request_id = self.generate_request_id()
        
        # 记录开始时间
        start_time = time.time()
        self.request_start_times[request_id] = start_time
        
        # 将model参数标准化
        if model is None or model == "":
            model = "default"
        
        # 检查model参数
        if model not in ["large", "small", "default"]:
            # 这里我们假设只有这三种特殊值，其他的都交给默认处理
            original_model = model
            model = "default"  # 将非标准值转换为default
            logging.info(f"[{request_id}] 检测到非标准模型名称: {original_model}，已转换为: {model}")
        
        # 获取endpoint
        endpoint = "chat/completions"
        if "endpoint" in request_data:
            endpoint = request_data.pop("endpoint")
        
        # 记录完整的请求内容(注意隐藏敏感信息)
        log_request = request_data.copy()
        if "api_key" in log_request:
            log_request["api_key"] = "*****"
        logging.info(f"[{request_id}] 客户端请求: stream={stream}, model={model}, endpoint={endpoint}")
        logging.info(f"[{request_id}] 请求内容: {json.dumps(log_request, ensure_ascii=False)}")
        
        # 记录用户输入更详细
        if "messages" in request_data:
            all_messages = []
            for i, msg in enumerate(request_data["messages"]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    all_messages.append(f"[{i+1}] {role}: {content}")
            
            if all_messages:
                logging.info(f"[{request_id}] 用户对话历史:\n" + "\n".join(all_messages))
        
        # 记录请求
        if self.logger:
            content_length = len(json.dumps(request_data))
            self.logger.log_request(request_id, model, stream, content_length)
        
        # 记录LLM池状态
        if self.logger:
            pool_status = self.llm_pool_manager.get_status()
            self.logger.log_llm_pool_status(
                request_id=request_id,
                large_pool_status=pool_status["large_pool"],
                small_pool_status=pool_status["small_pool"],
                queue_length=self.queue_manager.get_queue_length()
            )
            
            # 记录更详细的池状态
            large_pool_str = f"大模型池: {pool_status['large_pool']['current_load']}/{pool_status['large_pool']['total_capacity']} (已用/总容量)"
            small_pool_str = f"小模型池: {pool_status['small_pool']['current_load']}/{pool_status['small_pool']['total_capacity']} (已用/总容量)"
            queue_str = f"队列长度: {self.queue_manager.get_queue_length()}"
            
            # 记录每个实例状态
            large_instances = []
            for i, inst in enumerate(pool_status["large_pool"]["instances"]):
                status = "可用" if inst["available"] else "不可用"
                large_instances.append(f"  [{i+1}] {inst['model']} @ {inst['url']}: {inst['current_concurrency']}/{inst['max_concurrency']} ({status})")
            
            small_instances = []
            for i, inst in enumerate(pool_status["small_pool"]["instances"]):
                status = "可用" if inst["available"] else "不可用"
                small_instances.append(f"  [{i+1}] {inst['model']} @ {inst['url']}: {inst['current_concurrency']}/{inst['max_concurrency']} ({status})")
            
            pool_details = f"[{request_id}] LLM池详细状态:\n{large_pool_str}\n{small_pool_str}\n{queue_str}"
            if large_instances:
                pool_details += "\n大模型实例:\n" + "\n".join(large_instances)
            if small_instances:
                pool_details += "\n小模型实例:\n" + "\n".join(small_instances)
            
            logging.info(pool_details)
        
        # 获取目标池
        pool = self.llm_pool_manager.get_pool(model)
        
        try:
            # 检查池是否已满
            if pool.is_full():
                # 将请求放入队列
                queue_result = await self.queue_request(
                    request_id=request_id,
                    model_pool=model,
                    endpoint=endpoint,
                    request_data=request_data,
                    stream=stream,
                    timeout_seconds=timeout_seconds,
                    user_level=user_level
                )
                
                return queue_result
            
            # 直接处理请求
            return await self.process_request(
                request_id=request_id,
                model_pool=model,
                endpoint=endpoint,
                request_data=request_data,
                stream=stream
            )
            
        except Exception as e:
            error_msg = f"路由请求异常: {str(e)}"
            logging.error(f"[{request_id}] {error_msg}")
            
            if self.logger:
                self.logger.error(error_msg, request_id)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            return None, error_msg, processing_time
    
    async def process_request(
        self,
        request_id: str,
        model_pool: str,
        endpoint: str,
        request_data: Dict[str, Any],
        stream: bool = False
    ) -> Tuple[Any, Optional[str], float]:
        """
        处理请求，发送到LLM并获取响应
        
        Args:
            request_id: 请求ID
            model_pool: 模型池类型 ("large", "small" 或 "default")
            endpoint: API端点
            request_data: 请求数据
            stream: 是否使用流式响应
            
        Returns:
            (响应数据, 错误消息, 处理时间)
        """
        # 获取开始时间
        start_time = self.request_start_times.get(request_id, time.time())
        
        try:
            # 发送请求到LLM（带重试）
            response, error, llm_name = await self.llm_pool_manager.send_request_with_retry(
                request_id=request_id,
                pool_type=model_pool,
                endpoint=endpoint,
                payload=request_data,
                stream=stream,
                logger=self.logger
            )
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 记录请求完成
            if llm_name and self.logger:
                # 估算生成的令牌数
                tokens_generated = 0
                if response:
                    if stream:
                        # 流式响应，累计所有块的token数
                        for chunk in response:
                            if "choices" in chunk:
                                for choice in chunk["choices"]:
                                    if "delta" in choice and "content" in choice["delta"]:
                                        # 简单估算：假设每个词约1.5个token
                                        content = choice["delta"]["content"]
                                        if content:  # 确保content不是None
                                            tokens_generated += len(content.split()) * 1.5
                                            
                        # 记录流式响应的最后一部分内容
                        full_content = ""
                        for chunk in response:
                            if "choices" in chunk:
                                for choice in chunk["choices"]:
                                    if "delta" in choice and "content" in choice["delta"]:
                                        delta_content = choice["delta"].get("content", "")
                                        if delta_content:
                                            full_content += delta_content
                        if full_content:
                            logging.info(f"[{request_id}] 响应内容: {full_content}")
                    else:
                        # 非流式响应，从响应中获取token数
                        if "usage" in response and "completion_tokens" in response["usage"]:
                            tokens_generated = response["usage"]["completion_tokens"]
                        else:
                            # 简单估算
                            if "choices" in response:
                                for choice in response["choices"]:
                                    if "message" in choice and "content" in choice["message"]:
                                        content = choice["message"].get("content")
                                        if content:  # 确保content不是None
                                            tokens_generated += len(content.split()) * 1.5
                                        
                                        # 记录响应内容
                                        logging.info(f"[{request_id}] 响应内容: {content}")
                
                self.logger.log_completion(
                    request_id=request_id,
                    llm_name=llm_name,
                    processing_time=processing_time,
                    tokens_generated=int(tokens_generated)
                )
            
            # 清理请求记录
            if request_id in self.request_start_times:
                del self.request_start_times[request_id]
            
            return response, error, processing_time
            
        except Exception as e:
            error_msg = f"处理请求异常: {str(e)}"
            logging.error(f"[{request_id}] {error_msg}")
            
            if self.logger:
                self.logger.error(error_msg, request_id)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 清理请求记录
            if request_id in self.request_start_times:
                del self.request_start_times[request_id]
            
            return None, error_msg, processing_time
    
    async def queue_request(
        self,
        request_id: str,
        model_pool: str,
        endpoint: str,
        request_data: Dict[str, Any],
        stream: bool = False,
        timeout_seconds: Optional[int] = None,
        user_level: int = 0
    ) -> Tuple[Any, Optional[str], float]:
        """
        将请求放入等待队列
        
        Args:
            request_id: 请求ID
            model_pool: 模型池类型 ("large", "small" 或 "default")
            endpoint: API端点
            request_data: 请求数据
            stream: 是否使用流式响应
            timeout_seconds: 超时时间(秒)
            user_level: 用户等级
            
        Returns:
            (响应数据, 错误消息, 处理时间)
        """
        # 创建Future，用于在请求处理完成时返回结果
        result_future = asyncio.Future()
        
        # 创建完整的请求数据
        complete_request_data = {
            "endpoint": endpoint,
            "payload": request_data,
            "stream": stream
        }
        
        # 定义回调函数
        def request_callback(response_data, error):
            if not result_future.done():
                # 计算处理时间
                start_time = self.request_start_times.get(request_id, time.time())
                processing_time = time.time() - start_time
                
                if error:
                    result_future.set_result((None, error, processing_time))
                else:
                    result_future.set_result((response_data, None, processing_time))
        
        # 将请求加入队列
        success, message = self.queue_manager.enqueue_request(
            request_id=request_id,
            model_pool=model_pool,
            request_data=complete_request_data,
            user_level=user_level,
            timeout_seconds=timeout_seconds,
            callback=request_callback
        )
        
        if not success:
            # 队列添加失败
            error_msg = f"无法加入队列: {message}"
            logging.error(f"[{request_id}] {error_msg}")
            
            if self.logger:
                self.logger.error(error_msg, request_id)
            
            # 计算处理时间
            start_time = self.request_start_times.get(request_id, time.time())
            processing_time = time.time() - start_time
            
            # 清理请求记录
            if request_id in self.request_start_times:
                del self.request_start_times[request_id]
            
            return None, error_msg, processing_time
        
        if self.logger:
            self.logger.info(f"请求已加入队列: {message}", request_id)
        
        # 确保队列处理器正在运行
        if not self.queue_processor_task or self.queue_processor_task.done():
            self.start_queue_processor()
        
        # 等待结果
        try:
            response, error, processing_time = await result_future
            
            # 清理请求记录
            if request_id in self.request_start_times:
                del self.request_start_times[request_id]
            
            return response, error, processing_time
        except asyncio.CancelledError:
            # 请求被取消
            self.queue_manager.remove_request(request_id)
            
            error_msg = "请求已取消"
            logging.warning(f"[{request_id}] {error_msg}")
            
            if self.logger:
                self.logger.warning(error_msg, request_id)
            
            # 计算处理时间
            start_time = self.request_start_times.get(request_id, time.time())
            processing_time = time.time() - start_time
            
            # 清理请求记录
            if request_id in self.request_start_times:
                del self.request_start_times[request_id]
            
            return None, error_msg, processing_time
    
    def start_queue_processor(self) -> None:
        """
        启动队列处理器
        """
        if self.queue_processor_task and not self.queue_processor_task.done():
            return
        
        self.running = True
        self.queue_processor_task = asyncio.create_task(self._process_queue())
        logging.info("队列处理器已启动")
    
    async def _process_queue(self) -> None:
        """
        队列处理器，持续处理队列中的请求
        """
        while self.running:
            try:
                # 检查大模型池
                if not self.llm_pool_manager.large_pool.is_full():
                    # 尝试从队列获取请求
                    request = self.queue_manager.dequeue_request("large")
                    if request:
                        # 处理大模型请求
                        await self._handle_queued_request(request)
                        continue
                
                # 检查小模型池
                if not self.llm_pool_manager.small_pool.is_full():
                    # 尝试从队列获取请求
                    request = self.queue_manager.dequeue_request("small")
                    if request:
                        # 处理小模型请求
                        await self._handle_queued_request(request)
                        continue
                
                # 如果没有请求需要处理，等待一会儿
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logging.error(f"队列处理器异常: {str(e)}")
                await asyncio.sleep(1)
    
    async def _handle_queued_request(self, queued_request):
        """
        处理队列中的请求
        
        Args:
            queued_request: 队列请求对象
        """
        request_id = queued_request.request_id
        request_data = queued_request.request_data
        callback = queued_request.callback
        
        try:
            # 提取请求参数
            endpoint = request_data.get("endpoint", "chat/completions")
            payload = request_data.get("payload", {})
            stream = request_data.get("stream", False)
            
            # 计算队列等待时间
            queue_wait_time = time.time() - queued_request.arrival_time
            
            # 处理请求
            response, error, processing_time = await self.process_request(
                request_id=request_id,
                model_pool=queued_request.model_pool,
                endpoint=endpoint,
                request_data=payload,
                stream=stream
            )
            
            # 调用回调函数
            if callback:
                callback(response, error)
            
            # 记录完整处理信息
            if self.logger:
                self.logger.info(
                    f"队列请求处理完成，等待时间: {queue_wait_time:.2f}秒，处理时间: {processing_time:.2f}秒",
                    request_id
                )
            
        except Exception as e:
            error_msg = f"处理队列请求异常: {str(e)}"
            logging.error(f"[{request_id}] {error_msg}")
            
            if self.logger:
                self.logger.error(error_msg, request_id)
            
            # 调用回调函数报告错误
            if callback:
                callback(None, error_msg)
    
    async def stop(self) -> None:
        """
        停止路由决策器
        """
        self.running = False
        
        if self.queue_processor_task:
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
        
        logging.info("路由决策器已停止") 