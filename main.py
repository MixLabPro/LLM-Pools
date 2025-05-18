"""
启动LLM路由池管理系统的入口脚本
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入应用主模块
from src.app import main

if __name__ == "__main__":
    try:
        print("启动LLM路由池管理系统...")
        main()
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动失败: {str(e)}")
        sys.exit(1) 