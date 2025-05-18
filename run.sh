#!/bin/bash
echo "激活虚拟环境..."
source .env/bin/activate

echo "启动LLM路由池管理系统..."
python main.py 