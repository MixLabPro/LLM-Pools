import json
import os
import logging
from typing import Dict, List, Any, Optional

class ConfigManager:
    """
    配置管理器，负责读取和解析JSON配置文件
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.large_models: List[Dict[str, str]] = []
        self.small_models: List[Dict[str, str]] = []
        self.queue_settings: Dict[str, Any] = {}
        self.retry_settings: Dict[str, Any] = {}
        self.logging_settings: Dict[str, Any] = {}
        
        try:
            self.load_config()
            logging.info(f"配置已成功加载: {self.config_path}")
        except Exception as e:
            logging.error(f"加载配置失败: {str(e)}")
            raise
    
    def load_config(self) -> None:
        """
        加载配置文件
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            # 读取并解析JSON配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # 提取各部分配置
            self.large_models = self.config.get("large_models", [])
            self.small_models = self.config.get("small_models", [])
            self.queue_settings = self.config.get("queue_settings", {})
            self.retry_settings = self.config.get("retry_settings", {})
            self.logging_settings = self.config.get("logging", {})
            
            # 验证配置
            self._validate_config()
        except json.JSONDecodeError:
            logging.error(f"配置文件格式错误: {self.config_path}")
            raise
        except Exception as e:
            logging.error(f"加载配置出错: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """
        验证配置有效性
        """
        # 检查大模型和小模型配置
        if not self.large_models and not self.small_models:
            raise ValueError("配置中必须至少包含一个大模型或小模型")
        
        # 验证每个模型配置
        for model_list in [self.large_models, self.small_models]:
            for model in model_list:
                required_fields = ["url", "model", "api_key"]
                for field in required_fields:
                    if field not in model:
                        raise ValueError(f"模型配置缺少必要字段: {field}")
                
                # 验证request参数（如果有）
                if "request" in model:
                    try:
                        request_value = int(model["request"])
                        if request_value <= 0:
                            raise ValueError(f"request参数必须为正整数，当前值: {request_value}")
                    except (ValueError, TypeError):
                        raise ValueError(f"request参数必须为整数，当前值: {model['request']}")
    
    def get_model_pool(self, pool_type: str) -> List[Dict[str, str]]:
        """
        获取指定类型的模型池
        
        Args:
            pool_type: 模型池类型 ("large", "small" 或 "default")
            
        Returns:
            模型池配置列表
        """
        if pool_type == "large" or pool_type == "default":
            return self.large_models
        elif pool_type == "small":
            return self.small_models
        else:
            # 对于未知类型，返回默认模型池(大模型)
            logging.warning(f"未知的模型池类型: {pool_type}，使用默认模型池")
            return self.large_models
    
    def get_queue_timeout(self) -> int:
        """
        获取队列超时时间(秒)
        
        Returns:
            队列超时时间
        """
        return self.queue_settings.get("default_timeout", 30)
    
    def get_max_queue_length(self) -> int:
        """
        获取最大队列长度
        
        Returns:
            最大队列长度
        """
        return self.queue_settings.get("max_queue_length", 100)
    
    def get_retry_settings(self) -> Dict[str, Any]:
        """
        获取重试设置
        
        Returns:
            重试设置字典
        """
        return self.retry_settings
    
    def get_logging_settings(self) -> Dict[str, Any]:
        """
        获取日志设置
        
        Returns:
            日志设置字典
        """
        return self.logging_settings
    
    def reload_config(self) -> None:
        """
        重新加载配置文件
        """
        self.load_config()
        logging.info("配置已重新加载")

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_manager = ConfigManager()
    print(f"大模型: {len(config_manager.large_models)}")
    print(f"小模型: {len(config_manager.small_models)}")
    print(f"队列设置: {config_manager.queue_settings}")
    print(f"重试设置: {config_manager.retry_settings}") 