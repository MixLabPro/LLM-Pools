@echo off
echo 开始LLM池并发性能测试...
echo.

echo 第一阶段: 测试大模型池并发处理能力
echo 将发送5个大模型并发请求，验证大模型池并发处理能力
cd %~dp0
python src/test_llm_pool_client.py -c 5 -m large

echo.
echo 等待30秒后继续测试...
timeout /t 30 /nobreak > nul

echo.
echo 第二阶段: 测试小模型池并发处理能力
echo 将发送3个小模型并发请求，验证小模型池并发处理能力
python src/test_llm_pool_client.py -c 3 -m small

echo.
echo 等待30秒后继续测试...
timeout /t 30 /nobreak > nul

echo.
echo 第三阶段: 测试混合请求处理能力
echo 将发送5个混合类型并发请求，验证整体并发处理能力
python src/test_llm_pool_client.py -c 5

echo.
echo 测试完成! 