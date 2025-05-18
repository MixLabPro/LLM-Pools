import json
import time
import asyncio
import uuid
import logging
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, Request, Response, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

class OpenAICompatibleAPI:
    """
    提供与OpenAI API兼容的接口
    """
    
    def __init__(self, router_manager):
        """
        初始化API接口
        
        Args:
            router_manager: 路由决策器
        """
        self.router_manager = router_manager
        self.api = FastAPI(title="LLM Router API", description="OpenAI API Compatible LLM Router")
        
        # 注册路由
        self._register_routes()
    
    def _register_routes(self):
        """
        注册API路由
        """
        # 聊天完成接口
        @self.api.post("/v1/chat/completions")
        async def chat_completions(request: Request, background_tasks: BackgroundTasks):
            # 读取请求体
            body = await request.json()
            
            # 获取模型和流式参数
            model = body.get("model", "default")
            stream = body.get("stream", False)
            
            # 记录请求信息
            request_id = f"req-{uuid.uuid4().hex[:8]}"
            logging.info(f"[{request_id}] 收到/v1/chat/completions请求: model={model}, stream={stream}")
            
            # 打印请求内容
            if "messages" in body:
                messages_str = "\n  ".join([
                    f"[{i+1}] {msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    for i, msg in enumerate(body['messages'])
                    if msg.get('content')
                ])
                if messages_str:
                    logging.info(f"[{request_id}] 用户请求内容:\n  {messages_str}")
            
            # 记录其他请求参数
            other_params = {k: v for k, v in body.items() if k not in ["messages", "model", "stream"]}
            if other_params:
                logging.info(f"[{request_id}] 其他请求参数: {json.dumps(other_params, ensure_ascii=False)}")
            
            # 设置用户等级 (可以在后续版本中基于认证实现)
            user_level = 0
            
            # 路由请求
            if stream:
                # 使用流式响应
                logging.info(f"[{request_id}] 使用流式响应模式")
                return await self._stream_chat_response(body, model, user_level)
            else:
                # 使用非流式响应
                logging.info(f"[{request_id}] 使用非流式响应模式")
                return await self._regular_chat_response(body, model, user_level)
        
        # 文本完成接口
        @self.api.post("/v1/completions")
        async def completions(request: Request, background_tasks: BackgroundTasks):
            # 读取请求体
            body = await request.json()
            
            # 获取模型和流式参数
            model = body.get("model", "default")
            stream = body.get("stream", False)
            
            # 设置用户等级 (可以在后续版本中基于认证实现)
            user_level = 0
            
            # 对于文本完成，我们将请求转换为聊天格式
            if "prompt" in body:
                # 创建聊天格式的请求
                chat_body = body.copy()
                prompt = chat_body.pop("prompt")
                
                chat_body["messages"] = [
                    {"role": "user", "content": prompt}
                ]
                
                # 如果没有指定endpoint，设置为chat/completions
                chat_body["endpoint"] = "chat/completions"
                
                # 路由请求
                if stream:
                    # 使用流式响应
                    return await self._stream_chat_response(chat_body, model, user_level)
                else:
                    # 使用非流式响应
                    response = await self._regular_chat_response(chat_body, model, user_level)
                    
                    # 将chat格式的响应转换回completions格式
                    if "choices" in response and response["choices"]:
                        for choice in response["choices"]:
                            if "message" in choice and "content" in choice["message"]:
                                choice["text"] = choice["message"]["content"]
                                del choice["message"]
                    
                    return response
            else:
                # 如果没有prompt，直接路由请求
                if stream:
                    # 使用流式响应
                    return await self._stream_response(body, model, user_level)
                else:
                    # 使用非流式响应
                    return await self._regular_response(body, model, user_level)
        
        # 嵌入向量接口
        @self.api.post("/v1/embeddings")
        async def embeddings(request: Request):
            # 读取请求体
            body = await request.json()
            
            # 获取模型
            model = body.get("model", "default")
            
            # 设置用户等级 (可以在后续版本中基于认证实现)
            user_level = 0
            
            # 设置endpoint为embeddings
            body["endpoint"] = "embeddings"
            
            # 路由请求 (embeddings不支持流式响应)
            return await self._regular_response(body, model, user_level)
        
        # 健康检查接口
        @self.api.get("/health")
        async def health_check():
            # 获取当前状态
            status = self.router_manager.llm_pool_manager.get_status()
            
            # 计算各池可用实例数
            large_pool_available = sum(1 for inst in status["large_pool"]["instances"] if inst["available"])
            small_pool_available = sum(1 for inst in status["small_pool"]["instances"] if inst["available"])
            
            # 检查健康状态
            is_healthy = large_pool_available > 0 or small_pool_available > 0
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "large_pool": {
                    "total": len(status["large_pool"]["instances"]),
                    "available": large_pool_available,
                    "current_load": status["large_pool"]["current_load"],
                    "total_capacity": status["large_pool"]["total_capacity"]
                },
                "small_pool": {
                    "total": len(status["small_pool"]["instances"]),
                    "available": small_pool_available,
                    "current_load": status["small_pool"]["current_load"],
                    "total_capacity": status["small_pool"]["total_capacity"]
                },
                "queue_length": self.router_manager.queue_manager.get_queue_length()
            }
            
        # 添加/status端点与/health保持一致，提供兼容性
        @self.api.get("/status")
        async def status_check():
            return await health_check()
    
    async def _regular_response(self, request_data: Dict[str, Any], model: str, user_level: int = 0) -> Dict[str, Any]:
        """
        处理常规请求（非流式）
        
        Args:
            request_data: 请求数据
            model: 模型名称
            user_level: 用户等级
            
        Returns:
            响应数据
        """
        # 发送请求
        response, error, processing_time = await self.router_manager.route_request(
            request_data=request_data,
            model=model,
            stream=False,
            user_level=user_level
        )
        
        # 如果有错误，抛出异常
        if error:
            raise HTTPException(
                status_code=400,  # 始终使用400，不再特殊处理超时
                detail={"error": error}
            )
        
        # 返回响应
        return response
    
    async def _regular_chat_response(self, request_data: Dict[str, Any], model: str, user_level: int = 0) -> Dict[str, Any]:
        """
        处理聊天请求（非流式）
        
        Args:
            request_data: 请求数据
            model: 模型名称
            user_level: 用户等级
            
        Returns:
            响应数据
        """
        # 确保设置了endpoint
        request_data["endpoint"] = "chat/completions"
        
        # 发送请求
        return await self._regular_response(request_data, model, user_level)
    
    async def _stream_generator(self, request_data: Dict[str, Any], model: str, user_level: int = 0):
        """
        生成流式响应
        
        Args:
            request_data: 请求数据
            model: 模型名称
            user_level: 用户等级
            
        Yields:
            流式响应块
        """
        # 记录流式请求开始
        request_id = f"stream-{uuid.uuid4().hex[:8]}"
        logging.info(f"[{request_id}] 开始处理流式请求: model={model}")
        
        # 记录用户请求内容
        if "messages" in request_data:
            all_messages = []
            for i, msg in enumerate(request_data["messages"]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    all_messages.append(f"[{i+1}] {role}: {content}")
            
            if all_messages:
                logging.info(f"[{request_id}] 用户对话历史:\n" + "\n".join(all_messages))
        
        # 发送请求
        response, error, processing_time = await self.router_manager.route_request(
            request_data=request_data,
            model=model,
            stream=True,
            user_level=user_level
        )
        
        # 如果有错误，返回错误响应
        if error:
            error_response = {
                "error": {
                    "message": error,
                    "type": "router_error",
                    "code": "400"  # 始终使用400，不再特殊处理超时
                }
            }
            error_json = json.dumps(error_response)
            logging.error(f"[{request_id}] 流式请求错误: {error_json}")
            yield f"data: {error_json}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        # 逐个返回响应块
        chunk_count = 0
        full_content = ""
        
        for chunk in response:
            # 尝试将 chunk 对象序列化为 JSON 字符串
            # OpenAI 的 ChatCompletionChunk 对象通常是 Pydantic 模型，
            # 可以使用 .model_dump_json() 方法直接序列化为 JSON 字符串，
            # 或者使用 .model_dump() 转换为字典后再用 json.dumps()。
            # 我们优先使用 .model_dump_json()。
            try:
                # 尝试使用 Pydantic V2 的方法
                chunk_json = chunk.model_dump_json()
            except AttributeError:
                # 如果 model_dump_json 不存在，尝试 Pydantic V1 的 dict() 或其他转换为字典的方法
                try:
                    if hasattr(chunk, 'model_dump'): # Pydantic V2 model_dump
                        chunk_dict = chunk.model_dump()
                    elif hasattr(chunk, 'dict'): # Pydantic V1 dict()
                        chunk_dict = chunk.dict()
                    elif hasattr(chunk, 'to_dict'): # 一些库可能使用 to_dict()
                        chunk_dict = chunk.to_dict()
                    else: # 最后尝试直接转 __dict__，但这通常不适用于复杂的嵌套对象
                        chunk_dict = chunk.__dict__
                    chunk_json = json.dumps(chunk_dict)
                except Exception as e:
                    logging.error(f"[{request_id}] 无法序列化 chunk 对象: {chunk}. 错误: {e}")
                    # 如果序列化失败，发送一个错误信息给客户端并停止
                    error_payload = {
                        "error": {
                            "message": f"Failed to serialize response chunk: {type(chunk).__name__}",
                            "type": "serialization_error",
                            "code": "INTERNAL_SERVER_ERROR"
                        }
                    }
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            if chunk_count == 0:
                logging.info(f"[{request_id}] 开始返回流式响应，首个chunk: {chunk_json}")
            elif chunk_count % 10 == 0:  # 每10个chunk记录一次，避免日志过多
                logging.debug(f"[{request_id}] 流式响应进行中: 已返回{chunk_count}个chunks")
            
            # 收集完整内容用于日志记录
            if "choices" in chunk and len(chunk["choices"]) > 0:
                if "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]:
                    content = chunk["choices"][0]["delta"]["content"]
                    if content:
                        full_content += content
            
            # 确保每个块都正确格式化为SSE格式
            yield f"data: {chunk_json}\n\n"
            chunk_count += 1
            
            # 添加适当的延迟以确保客户端能够接收流式响应
            await asyncio.sleep(0.01)
        
        # 记录最后一个chunks和完整内容
        if chunk_count > 0:
            logging.info(f"[{request_id}] 流式响应完成: 总共返回{chunk_count}个chunks，耗时{processing_time:.2f}秒")
            if full_content:
                logging.info(f"[{request_id}] 完整响应内容: {full_content}")
        
        # 结束标志
        yield "data: [DONE]\n\n"
    
    async def _stream_response(self, request_data: Dict[str, Any], model: str, user_level: int = 0) -> StreamingResponse:
        """
        处理流式请求
        
        Args:
            request_data: 请求数据
            model: 模型名称
            user_level: 用户等级
            
        Returns:
            流式响应
        """
        # 确保请求中标记了stream=True
        request_data["stream"] = True
        
        # 记录完整请求内容
        log_request = request_data.copy()
        if "api_key" in log_request:
            log_request["api_key"] = "*****"
        logging.info(f"流式请求内容: {json.dumps(log_request, ensure_ascii=False)}")
        
        # 返回流式响应，设置正确的headers
        return StreamingResponse(
            self._stream_generator(request_data, model, user_level),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )
    
    async def _stream_chat_response(self, request_data: Dict[str, Any], model: str, user_level: int = 0) -> StreamingResponse:
        """
        处理聊天流式请求
        
        Args:
            request_data: 请求数据
            model: 模型名称
            user_level: 用户等级
            
        Returns:
            流式响应
        """
        # 确保设置了endpoint
        request_data["endpoint"] = "chat/completions"
        
        # 确保请求中标记了stream=True
        request_data["stream"] = True
        
        # 返回流式响应
        return await self._stream_response(request_data, model, user_level)
    
    def get_app(self) -> FastAPI:
        """
        获取FastAPI应用
        
        Returns:
            FastAPI应用
        """
        return self.api 