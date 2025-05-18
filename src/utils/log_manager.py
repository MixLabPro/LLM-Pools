import os
import logging
import time
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from typing import Dict, Any, Optional

class LogManager:
    """
    日志管理器，处理所有系统日志的记录和输出
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化日志管理器
        
        Args:
            config: 日志配置字典
        """
        self.config = config
        self.logger = logging.getLogger("llm_router")
        
        # 设置日志级别
        log_level = getattr(logging, config.get("level", "INFO").upper())
        self.logger.setLevel(log_level)
        
        # 创建日志目录
        log_file_path = config.get("file_path", "./logs/router.log")
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # 配置控制台日志处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        console_handler.setFormatter(console_format)
        
        # 配置文件日志处理器 (JSON格式，支持轮转)
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=config.get("rotate_size_mb", 10) * 1024 * 1024,
            backupCount=config.get("keep_logs_days", 7)
        )
        file_handler.setLevel(log_level)
        
        # 创建JSON格式化器
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(message)s %(request_id)s',
            rename_fields={'levelname': 'level', 'asctime': 'timestamp'}
        )
        file_handler.setFormatter(json_formatter)
        
        # 添加处理器到logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # 记录初始化完成
        self.info("日志系统初始化完成", None)
    
    def _log(self, level: int, message: str, request_id: Optional[str] = None, **kwargs) -> None:
        """
        记录日志的通用方法
        
        Args:
            level: 日志级别
            message: 日志消息
            request_id: 请求ID (可选)
            **kwargs: 其他日志字段
        """
        extra = {"request_id": request_id if request_id else ""}
        if kwargs:
            extra.update(kwargs)
        
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, request_id: Optional[str] = None, **kwargs) -> None:
        """
        记录INFO级别日志
        
        Args:
            message: 日志消息
            request_id: 请求ID (可选)
            **kwargs: 其他日志字段
        """
        self._log(logging.INFO, message, request_id, **kwargs)
    
    def error(self, message: str, request_id: Optional[str] = None, **kwargs) -> None:
        """
        记录ERROR级别日志
        
        Args:
            message: 日志消息
            request_id: 请求ID (可选)
            **kwargs: 其他日志字段
        """
        self._log(logging.ERROR, message, request_id, **kwargs)
    
    def warning(self, message: str, request_id: Optional[str] = None, **kwargs) -> None:
        """
        记录WARNING级别日志
        
        Args:
            message: 日志消息
            request_id: 请求ID (可选)
            **kwargs: 其他日志字段
        """
        self._log(logging.WARNING, message, request_id, **kwargs)
    
    def debug(self, message: str, request_id: Optional[str] = None, **kwargs) -> None:
        """
        记录DEBUG级别日志
        
        Args:
            message: 日志消息
            request_id: 请求ID (可选)
            **kwargs: 其他日志字段
        """
        self._log(logging.DEBUG, message, request_id, **kwargs)
    
    def log_request(self, request_id: str, model: str, stream: bool, content_length: int) -> None:
        """
        记录请求日志
        
        Args:
            request_id: 请求ID
            model: 模型名称
            stream: 是否为流式请求
            content_length: 内容长度
        """
        self.info(
            f"新请求已接收: ",
            request_id,
            model=model,
            stream=stream,
            content_length=content_length,
            timestamp=time.time()
        )
    
    def log_llm_pool_status(
        self, 
        request_id: str, 
        large_pool_status: Dict[str, Any],
        small_pool_status: Dict[str, Any],
        queue_length: int
    ) -> None:
        """
        记录LLM池状态日志
        
        Args:
            request_id: 请求ID
            large_pool_status: 大模型池状态
            small_pool_status: 小模型池状态
            queue_length: 队列长度
        """
        self.info(
            f"当前LLM池状态: ",
            request_id,
            large_pool=large_pool_status,
            small_pool=small_pool_status,
            queue_length=queue_length
        )
    
    def log_route_decision(
        self,
        request_id: str,
        selected_llm: str,
        concurrency: int,
        max_concurrency: int,
        reason: str
    ) -> None:
        """
        记录路由决策日志
        
        Args:
            request_id: 请求ID
            selected_llm: 选择的LLM
            concurrency: 当前并发数
            max_concurrency: 最大并发数
            reason: 决策原因
        """
        self.info(
            f"路由决策: ",
            request_id,
            selected_llm=selected_llm,
            concurrency=f"{concurrency}/{max_concurrency}",
            reason=reason
        )
    
    def log_error_retry(
        self,
        request_id: str,
        llm_name: str,
        error_message: str,
        retry_attempt: int,
        max_retries: int
    ) -> None:
        """
        记录错误和重试日志
        
        Args:
            request_id: 请求ID
            llm_name: LLM名称
            error_message: 错误消息
            retry_attempt: 重试次数
            max_retries: 最大重试次数
        """
        self.error(
            f"LLM请求失败: ",
            request_id,
            llm=llm_name,
            error=error_message,
            retry_attempt=f"{retry_attempt}/{max_retries}"
        )
    
    def log_completion(
        self,
        request_id: str,
        llm_name: str,
        processing_time: float,
        tokens_generated: int,
        queue_wait_time: float = 0
    ) -> None:
        """
        记录请求完成日志
        
        Args:
            request_id: 请求ID
            llm_name: 使用的LLM
            processing_time: 处理时间(秒)
            tokens_generated: 生成的令牌数
            queue_wait_time: 队列等待时间(秒)
        """
        self.info(
            f"请求完成: ",
            request_id,
            llm=llm_name,
            processing_time=f"{processing_time:.2f}秒",
            tokens=tokens_generated,
            queue_wait_time=f"{queue_wait_time:.2f}秒" if queue_wait_time > 0 else "0秒"
        ) 