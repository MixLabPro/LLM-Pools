@echo off
echo 激活虚拟环境...
call .env\Scripts\activate

echo 启动LLM路由池管理系统...
python main.py

pause 