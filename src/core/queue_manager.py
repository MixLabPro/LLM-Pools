import time
import uuid
import heapq
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
import logging

@dataclass(order=True)
class QueuedRequest:
    """
    队列中的请求对象
    """
    # 优先级（数字越小优先级越高）
    priority: int = field(compare=True)
    
    # 请求到达时间
    arrival_time: float = field(compare=False)
    
    # 请求超时时间点
    timeout_time: float = field(compare=False)
    
    # 请求ID
    request_id: str = field(compare=False)
    
    # 请求的目标模型池
    model_pool: str = field(compare=False)
    
    # 请求内容
    request_data: Dict[str, Any] = field(compare=False)
    
    # 回调函数，用于通知队列处理完毕
    callback: Callable = field(compare=False, default=None)
    
    # 用户等级
    user_level: int = field(compare=False, default=0)

class QueueManager:
    """
    队列管理器，管理请求等待队列和超时控制
    """
    
    def __init__(self, max_queue_length: int = 100, default_timeout_seconds: int = 30):
        """
        初始化队列管理器
        
        Args:
            max_queue_length: 最大队列长度
            default_timeout_seconds: 默认请求超时时间(秒)
        """
        self.max_queue_length = max_queue_length
        self.default_timeout = default_timeout_seconds
        
        # 优先级队列
        self.request_queue: List[QueuedRequest] = []
        
        # 队列锁，确保线程安全
        self.queue_lock = threading.Lock()
        
        # 超时检查线程
        self.timeout_checker_thread = threading.Thread(
            target=self._timeout_checker,
            daemon=True,
            name="timeout-checker"
        )
        self.running = True
        self.timeout_checker_thread.start()
        
        logging.info(f"队列管理器已初始化，最大长度: {max_queue_length}, 默认超时: {default_timeout_seconds}秒")
    
    def enqueue_request(
        self,
        request_id: str,
        model_pool: str,
        request_data: Dict[str, Any],
        user_level: int = 0,
        timeout_seconds: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> Tuple[bool, str]:
        """
        将请求加入队列
        
        Args:
            request_id: 请求ID
            model_pool: 模型池类型 ("large", "small")
            request_data: 请求数据
            user_level: 用户等级 (0-10, 数字越大优先级越高)
            timeout_seconds: 超时时间(秒)，如果为None则使用默认值
            callback: 回调函数，在请求被处理或超时时调用
            
        Returns:
            (成功/失败, 错误消息)
        """
        with self.queue_lock:
            # 检查队列是否已满
            if len(self.request_queue) >= self.max_queue_length:
                return False, "队列已满，请稍后重试"
            
            # 设置超时时间
            timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout
            arrival_time = time.time()
            timeout_time = arrival_time + timeout
            
            # 计算优先级（基于用户等级）
            # 优先级数字越小越优先处理
            priority = 100 - (user_level * 10)  # 用户等级0-10对应优先级100-0
            
            # 创建队列请求对象
            queued_request = QueuedRequest(
                priority=priority,
                arrival_time=arrival_time,
                timeout_time=timeout_time,
                request_id=request_id,
                model_pool=model_pool,
                request_data=request_data,
                callback=callback,
                user_level=user_level
            )
            
            # 加入优先级队列
            heapq.heappush(self.request_queue, queued_request)
            
            # 计算队列位置
            queue_position = self._get_position(request_id)
            
            logging.info(f"请求 {request_id} 已加入队列, 位置: {queue_position}, 优先级: {priority}, 超时: {timeout}秒")
            
            return True, f"请求已加入队列，位置: {queue_position}"
    
    def dequeue_request(self, model_pool: str) -> Optional[QueuedRequest]:
        """
        从队列中取出最高优先级的请求
        
        Args:
            model_pool: 模型池类型 ("large", "small")
            
        Returns:
            队列请求对象，如果队列为空则返回None
        """
        with self.queue_lock:
            if not self.request_queue:
                return None
            
            # 首先尝试取出指定模型池的请求
            for i, request in enumerate(self.request_queue):
                if request.model_pool == model_pool:
                    # 移除并返回找到的请求
                    request = self.request_queue.pop(i)
                    # 重建堆
                    heapq.heapify(self.request_queue)
                    logging.info(f"请求 {request.request_id} 已从队列中取出，等待时间: {time.time() - request.arrival_time:.2f}秒")
                    return request
            
            # 如果没有找到指定模型池的请求，返回None
            return None
    
    def get_queue_length(self) -> int:
        """
        获取当前队列长度
        
        Returns:
            队列中的请求数量
        """
        with self.queue_lock:
            return len(self.request_queue)
    
    def _get_position(self, request_id: str) -> int:
        """
        获取请求在队列中的位置
        
        Args:
            request_id: 请求ID
            
        Returns:
            队列位置 (从1开始)，如果未找到则返回-1
        """
        for i, req in enumerate(sorted(self.request_queue)):
            if req.request_id == request_id:
                return i + 1
        return -1
    
    def remove_request(self, request_id: str) -> bool:
        """
        从队列中移除指定请求
        
        Args:
            request_id: 请求ID
            
        Returns:
            是否成功移除
        """
        with self.queue_lock:
            for i, req in enumerate(self.request_queue):
                if req.request_id == request_id:
                    removed = self.request_queue.pop(i)
                    heapq.heapify(self.request_queue)
                    logging.info(f"请求 {request_id} 已从队列中移除")
                    
                    # 调用回调函数，表示请求已被移除
                    if removed.callback:
                        try:
                            removed.callback(None, "请求已被移除")
                        except Exception as e:
                            logging.error(f"回调函数执行失败: {str(e)}")
                    
                    return True
            
            return False
    
    def _timeout_checker(self) -> None:
        """
        超时检查线程，定期检查并移除超时的请求
        """
        while self.running:
            try:
                with self.queue_lock:
                    current_time = time.time()
                    timed_out_indices = []
                    
                    # 找出所有已超时的请求
                    for i, req in enumerate(self.request_queue):
                        if current_time > req.timeout_time:
                            timed_out_indices.append(i)
                    
                    # 从后向前移除超时请求（避免索引变化问题）
                    for i in sorted(timed_out_indices, reverse=True):
                        timed_out_req = self.request_queue.pop(i)
                        logging.warning(f"请求 {timed_out_req.request_id} 已超时，从队列中移除")
                        
                        # 调用回调函数，报告超时
                        if timed_out_req.callback:
                            try:
                                timed_out_req.callback(None, "请求超时")
                            except Exception as e:
                                logging.error(f"超时回调执行失败: {str(e)}")
                    
                    # 如果移除了请求，需要重新构建堆
                    if timed_out_indices:
                        heapq.heapify(self.request_queue)
            
            except Exception as e:
                logging.error(f"超时检查线程错误: {str(e)}")
            
            # 间隔一段时间再次检查
            time.sleep(1)
    
    def shutdown(self) -> None:
        """
        关闭队列管理器，停止超时检查线程
        """
        self.running = False
        if self.timeout_checker_thread.is_alive():
            self.timeout_checker_thread.join(timeout=5)
        logging.info("队列管理器已关闭")

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建队列管理器
    queue_mgr = QueueManager(max_queue_length=5, default_timeout_seconds=10)
    
    # 模拟回调函数
    def callback(data, error):
        print(f"回调: 数据={data}, 错误={error}")
    
    # 添加测试请求
    for i in range(3):
        success, msg = queue_mgr.enqueue_request(
            request_id=f"req-{i}",
            model_pool="large" if i % 2 == 0 else "small",
            request_data={"test": f"data-{i}"},
            user_level=i,
            callback=callback
        )
        print(f"请求 {i} 入队: {success}, {msg}")
    
    # 获取队列长度
    print(f"队列长度: {queue_mgr.get_queue_length()}")
    
    # 取出请求
    req = queue_mgr.dequeue_request("large")
    if req:
        print(f"取出请求: {req.request_id}, 数据: {req.request_data}")
    
    # 等待超时
    print("等待请求超时...")
    time.sleep(12)
    
    # 再次获取队列长度
    print(f"超时后队列长度: {queue_mgr.get_queue_length()}")
    
    # 关闭队列管理器
    queue_mgr.shutdown() 