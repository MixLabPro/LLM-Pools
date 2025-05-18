import os
import asyncio
import logging
import json
import uvicorn
from dotenv import load_dotenv
from typing import Dict, Any

from src.core.config_manager import ConfigManager
from src.core.llm_pool import LLMPoolManager
from src.core.queue_manager import QueueManager
from src.core.router import RouterManager
from src.utils.log_manager import LogManager
from src.api.openai_compatible import OpenAICompatibleAPI

class LLMRouterApp:
    """
    LLM路由池应用程序
    """
    
    def __init__(self, config_path: str = "config.json", env_path: str = "config.env"):
        """
        初始化LLM路由池应用程序
        
        Args:
            config_path: 配置文件路径
            env_path: 环境变量文件路径
        """
        # 加载环境变量
        load_dotenv(env_path)
        
        # 初始化配置管理器
        self.config_manager = ConfigManager(config_path)
        
        # 初始化日志管理器
        self.log_manager = LogManager(self.config_manager.get_logging_settings())
        
        # 初始化LLM池管理器，使用默认值3作为没有request参数时的备选值
        self.llm_pool_manager = LLMPoolManager(max_concurrency_per_llm=3)
        
        # 从配置添加模型
        self.llm_pool_manager.add_models_from_config(
            large_models=self.config_manager.large_models,
            small_models=self.config_manager.small_models
        )
        
        # 设置重试参数
        retry_settings = self.config_manager.get_retry_settings()
        self.llm_pool_manager.set_retry_settings(
            max_retries=retry_settings.get("max_retries", 3),
            retry_delay_ms=retry_settings.get("retry_delay_ms", 100)
        )
        
        # 初始化队列管理器
        self.queue_manager = QueueManager(
            max_queue_length=self.config_manager.get_max_queue_length(),
            default_timeout_seconds=self.config_manager.get_queue_timeout()
        )
        
        # 初始化路由决策器
        self.router_manager = RouterManager(
            llm_pool_manager=self.llm_pool_manager,
            queue_manager=self.queue_manager,
            logger=self.log_manager
        )
        
        # 初始化API接口
        self.api = OpenAICompatibleAPI(self.router_manager)
        
        self.log_manager.info("LLM路由池应用程序已初始化", None)
    
    async def start(self):
        """
        启动应用程序
        """
        try:
            # 启动健康检查
            await self.llm_pool_manager.start_health_checks()
            
            # 启动队列处理器
            self.router_manager.start_queue_processor()
            
            # 打印详细的池状态
            status = self.llm_pool_manager.get_status()
            
            # 大模型池状态
            large_stats = status["large_pool"]
            large_instances_str = "\n    ".join([
                f"{i+1}. {inst['model']} @ {inst['url']}: {inst['current_concurrency']}/{inst['max_concurrency']} 并发 (状态: {'可用' if inst['available'] else '不可用'})"
                for i, inst in enumerate(large_stats["instances"])
            ])
            large_pool_info = (
                f"大模型池 ({large_stats['utilization_percent']}% 使用率):\n"
                f"  实例数: {large_stats['instance_count']} (可用: {large_stats['available_instance_count']})\n"
                f"  负载: {large_stats['current_load']}/{large_stats['total_capacity']} (使用/总容量)\n"
                f"  模型列表:\n    {large_instances_str if large_instances_str else '无实例'}"
            )
            
            # 小模型池状态
            small_stats = status["small_pool"]
            small_instances_str = "\n    ".join([
                f"{i+1}. {inst['model']} @ {inst['url']}: {inst['current_concurrency']}/{inst['max_concurrency']} 并发 (状态: {'可用' if inst['available'] else '不可用'})"
                for i, inst in enumerate(small_stats["instances"])
            ])
            small_pool_info = (
                f"小模型池 ({small_stats['utilization_percent']}% 使用率):\n"
                f"  实例数: {small_stats['instance_count']} (可用: {small_stats['available_instance_count']})\n"
                f"  负载: {small_stats['current_load']}/{small_stats['total_capacity']} (使用/总容量)\n"
                f"  模型列表:\n    {small_instances_str if small_instances_str else '无实例'}"
            )
            
            # 总体状态
            overall_stats = status["overall"]
            overall_info = (
                f"池总体状态 ({overall_stats['overall_utilization_percent']}% 总使用率):\n"
                f"  总实例数: {overall_stats['total_instances']} (可用: {overall_stats['available_instances']})\n"
                f"  总负载: {overall_stats['current_load']}/{overall_stats['total_capacity']} (使用/总容量)"
            )
            
            # 打印状态信息
            status_info = f"\n\n=== LLM池状态 ===\n{large_pool_info}\n\n{small_pool_info}\n\n{overall_info}\n"
            logging.info(status_info)
            print(status_info)
            
            # 启动定期状态更新任务
            self.start_status_reporting()
            
            self.log_manager.info("LLM路由池应用程序已启动", None)
            
        except Exception as e:
            self.log_manager.error(f"启动应用程序失败: {str(e)}", None)
            raise
    
    def start_status_reporting(self, interval_seconds: int = 60):
        """
        启动定期状态报告任务
        
        Args:
            interval_seconds: 报告间隔时间(秒)
        """
        async def status_reporting_task():
            try:
                while True:
                    # 等待指定间隔
                    await asyncio.sleep(interval_seconds)
                    
                    # 获取并记录状态
                    status = self.llm_pool_manager.get_status()
                    
                    # 简化状态报告
                    large_pool = status["large_pool"]
                    small_pool = status["small_pool"]
                    overall = status["overall"]
                    
                    status_msg = (
                        f"=== LLM池状态更新 ===\n"
                        f"大模型池: {large_pool['current_load']}/{large_pool['total_capacity']} ({large_pool['utilization_percent']}%), "
                        f"实例: {large_pool['available_instance_count']}/{large_pool['instance_count']} 可用\n"
                        f"小模型池: {small_pool['current_load']}/{small_pool['total_capacity']} ({small_pool['utilization_percent']}%), "
                        f"实例: {small_pool['available_instance_count']}/{small_pool['instance_count']} 可用\n"
                        f"总体: {overall['current_load']}/{overall['total_capacity']} ({overall['overall_utilization_percent']}%), "
                        f"实例: {overall['available_instances']}/{overall['total_instances']} 可用"
                    )
                    
                    logging.info(status_msg)
                    
            except asyncio.CancelledError:
                logging.info("状态报告任务已取消")
            except Exception as e:
                logging.error(f"状态报告任务异常: {str(e)}")
        
        # 创建任务
        self.status_reporting_task = asyncio.create_task(status_reporting_task())
        logging.info(f"定期状态报告任务已启动，间隔: {interval_seconds}秒")
    
    async def stop(self):
        """
        停止应用程序
        """
        try:
            # 停止状态报告任务
            if hasattr(self, 'status_reporting_task') and self.status_reporting_task:
                self.status_reporting_task.cancel()
                try:
                    await self.status_reporting_task
                except asyncio.CancelledError:
                    pass
            
            # 停止路由决策器
            await self.router_manager.stop()
            
            # 关闭LLM池
            await self.llm_pool_manager.close()
            
            # 关闭队列管理器
            self.queue_manager.shutdown()
            
            self.log_manager.info("LLM路由池应用程序已停止", None)
            
        except Exception as e:
            self.log_manager.error(f"停止应用程序失败: {str(e)}", None)
            raise

# 创建全局应用实例
app_instance = None

def get_app():
    """
    获取应用实例
    
    Returns:
        应用实例
    """
    global app_instance
    
    if app_instance is None:
        app_instance = LLMRouterApp()
    
    return app_instance

# FastAPI应用
app = get_app().api.get_app()

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    await get_app().start()

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    await get_app().stop()

def main():
    """主函数"""
    # 从环境变量获取服务器配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # 运行服务器
    uvicorn.run(
        "src.app:app",
        host=host,
        port=port,
        reload=False
    )

if __name__ == "__main__":
    main() 