#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import time
import uuid
import json
import logging
import random
from typing import Dict, Any, List
import openai
import httpx
from tqdm import tqdm
import prettytable
from datetime import datetime
import argparse
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 测试配置
SERVER_URL = "http://localhost:8000"  # 服务器地址
API_KEY = "test-api-key"  # API密钥
CONCURRENT_REQUESTS = 6  # 并发请求数
REQUEST_TIMEOUT = 30  # 请求超时时间(秒)

# 测试提示词列表
TEST_PROMPTS = [
    "给我讲个笑话",
    "Python如何读取文件？",
    "请解释量子力学的基本原理",
    "写一首关于春天的诗",
    "列出5种常见的排序算法",
    "如何制作披萨？",
    "解释递归的概念",
    "什么是区块链技术？",
    "给我一个简单的健身计划",
    "如何学习一门新语言？"
]

class RequestStats:
    """请求统计信息类"""
    
    def __init__(self):
        # 总体统计
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.total_response_time = 0
        
        # 每个LLM的统计
        self.llm_stats = {}
        
        # 进行中的请求
        self.in_progress = {}
        
        # 锁
        self.lock = asyncio.Lock()
    
    async def start_request(self, request_id: str):
        """记录请求开始"""
        async with self.lock:
            self.total_requests += 1
            self.in_progress[request_id] = {
                "start_time": time.time(),
                "status": "进行中"
            }
    
    async def complete_request(self, request_id: str, llm_name: str, response_time: float, tokens: int = 0):
        """记录请求完成"""
        async with self.lock:
            self.completed_requests += 1
            self.total_tokens += tokens
            self.total_response_time += response_time
            
            if request_id in self.in_progress:
                del self.in_progress[request_id]
            
            # 更新LLM统计
            if llm_name not in self.llm_stats:
                self.llm_stats[llm_name] = {
                    "requests": 0,
                    "tokens": 0,
                    "total_time": 0
                }
            
            self.llm_stats[llm_name]["requests"] += 1
            self.llm_stats[llm_name]["tokens"] += tokens
            self.llm_stats[llm_name]["total_time"] += response_time
    
    async def fail_request(self, request_id: str, error: str):
        """记录请求失败"""
        async with self.lock:
            self.failed_requests += 1
            
            if request_id in self.in_progress:
                self.in_progress[request_id]["status"] = f"失败: {error}"
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        avg_response_time = self.total_response_time / max(1, self.completed_requests)
        
        return {
            "总请求数": self.total_requests,
            "已完成": self.completed_requests,
            "失败": self.failed_requests,
            "进行中": len(self.in_progress),
            "平均响应时间": f"{avg_response_time:.2f}秒",
            "总生成Token": self.total_tokens,
            "LLM使用情况": self.llm_stats
        }
    
    def get_in_progress(self) -> Dict[str, Any]:
        """获取进行中的请求"""
        return self.in_progress

async def send_request(request_id: str, prompt: str, model: str, stats: RequestStats) -> None:
    """
    发送请求到LLM池服务器
    
    Args:
        request_id: 请求ID
        prompt: 提示词
        model: 模型类型 ("large", "small", "default")
        stats: 统计对象
    """
    try:
        # 记录请求开始
        await stats.start_request(request_id)
        
        # 使用OpenAI SDK创建客户端
        client = openai.AsyncOpenAI(
            api_key=API_KEY,
            base_url=f"{SERVER_URL}/v1",
            timeout=REQUEST_TIMEOUT
        )
        
        # 构造请求数据
        messages = [{"role": "user", "content": prompt}]
        
        # 发送请求
        start_time = time.time()
        
        try:
            # 使用标准的OpenAI SDK调用
            logging.info(f"[{request_id}] 开始发送请求 (模型: {model}, 提示词: {prompt[:30]}...)")
            
            response = await client.chat.completions.create(
                model=model,  # 使用指定的模型
                messages=messages,
                stream=False,
                user=request_id  # 使用user字段作为请求标识符
            )
            
            response_time = time.time() - start_time
            
            # 解析响应
            content = ""
            tokens = 0
            llm_name = "未知LLM"
            
            # 从响应中获取信息
            if hasattr(response, 'usage') and response.usage:
                tokens = getattr(response.usage, 'total_tokens', 0)
            
            # 尝试从自定义字段获取LLM信息
            if hasattr(response, 'llm_info'):
                llm_name = response.llm_info
            else:
                # 尝试从模型名称获取信息
                llm_name = getattr(response, 'model', '未知LLM')
            
            # 获取生成的内容
            if hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message'):
                    content = getattr(response.choices[0].message, 'content', '')
            
            # 记录请求完成
            await stats.complete_request(request_id, llm_name, response_time, tokens)
            
            logging.info(f"[{request_id}] 请求完成: 使用LLM={llm_name}, 耗时: {response_time:.2f}秒, Tokens: {tokens}")
            
            # 仅打印内容的前30个字符
            content_preview = content[:30] + "..." if len(content) > 30 else content
            logging.info(f"[{request_id}] 响应内容: {content_preview}")
                
        except Exception as e:
            error = f"OpenAI API调用失败: {str(e)}"
            await stats.fail_request(request_id, error)
            logging.error(f"[{request_id}] {error}")
            # 失败后不重试
    
    except Exception as e:
        error = f"请求异常: {str(e)}"
        await stats.fail_request(request_id, error)
        logging.error(f"[{request_id}] {error}")
        # 失败后不重试

async def print_live_stats(stats: RequestStats, total_requests: int):
    """打印实时统计信息"""
    progress_bar = tqdm(total=total_requests, desc="请求进度")
    last_completed = 0
    
    while stats.completed_requests + stats.failed_requests < total_requests:
        # 更新进度条
        current_completed = stats.completed_requests
        progress_bar.update(current_completed - last_completed)
        last_completed = current_completed
        
        # 创建一个表格
        summary = stats.get_summary()
        
        table = prettytable.PrettyTable()
        table.field_names = ["指标", "值"]
        table.align["指标"] = "l"
        table.align["值"] = "l"
        
        table.add_row(["总请求数", summary["总请求数"]])
        table.add_row(["已完成", summary["已完成"]])
        table.add_row(["失败", summary["失败"]])
        table.add_row(["进行中", summary["进行中"]])
        table.add_row(["平均响应时间", summary["平均响应时间"]])
        
        # LLM使用情况表格
        llm_table = prettytable.PrettyTable()
        llm_table.field_names = ["LLM名称", "请求数", "平均响应时间(秒)"]
        
        for llm_name, llm_data in summary["LLM使用情况"].items():
            avg_time = llm_data["total_time"] / max(1, llm_data["requests"])
            llm_table.add_row([
                llm_name, 
                llm_data["requests"],
                f"{avg_time:.2f}"
            ])
        
        # 打印统计信息
        print("\n" + "="*50)
        print(f"测试统计信息 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(table)
        print("\nLLM使用情况:")
        print(llm_table)
        print("="*50 + "\n")
        
        # 等待一段时间
        await asyncio.sleep(1)
    
    # 完成进度条
    progress_bar.update(total_requests - last_completed)
    progress_bar.close()

async def print_final_stats(stats: RequestStats):
    """打印最终统计信息"""
    summary = stats.get_summary()
    
    print("\n" + "="*50)
    print("测试完成! 最终统计结果:")
    print("="*50)
    
    # 基本统计
    basic_table = prettytable.PrettyTable()
    basic_table.field_names = ["指标", "值"]
    basic_table.align["指标"] = "l"
    basic_table.align["值"] = "l"
    
    avg_time = summary.get("平均响应时间", "未知")
    success_rate = (summary["已完成"] / max(1, summary["总请求数"])) * 100
    
    basic_table.add_row(["总请求数", summary["总请求数"]])
    basic_table.add_row(["成功请求", summary["已完成"]])
    basic_table.add_row(["失败请求", summary["失败"]])
    basic_table.add_row(["成功率", f"{success_rate:.2f}%"])
    basic_table.add_row(["平均响应时间", avg_time])
    basic_table.add_row(["总生成Token", summary["总生成Token"]])
    
    print(basic_table)
    
    # LLM统计
    llm_table = prettytable.PrettyTable()
    llm_table.field_names = ["LLM名称", "请求数", "请求占比", "平均响应时间(秒)", "Token数"]
    
    for llm_name, llm_data in summary["LLM使用情况"].items():
        avg_time = llm_data["total_time"] / max(1, llm_data["requests"])
        request_percentage = (llm_data["requests"] / max(1, summary["已完成"])) * 100
        
        llm_table.add_row([
            llm_name, 
            llm_data["requests"],
            f"{request_percentage:.2f}%",
            f"{avg_time:.2f}",
            llm_data["tokens"]
        ])
    
    print("\nLLM使用情况:")
    print(llm_table)
    print("="*50)

async def get_server_status():
    """获取服务器状态"""
    try:
        # 使用 httpx 直接发送异步请求
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SERVER_URL}/status", timeout=5)
            # 记录HTTP请求的日志
            logging.info(f'HTTP请求: GET {SERVER_URL}/status "{response.status_code} {response.reason_phrase}"')
            if response.status_code == 200:
                # 尝试解析JSON，并捕获可能的解析错误
                try:
                    return response.json()
                except json.JSONDecodeError as je:
                    logging.error(f"服务器状态JSON解析失败: {str(je)}, 响应内容: {response.text}")
                    return None # 或者返回一个表示错误状态的特定对象
            else:
                # 如果状态码不是200，也记录错误
                logging.error(f"获取服务器状态失败，状态码: {response.status_code}, 内容: {response.text}")
    except httpx.RequestError as e: # 更具体的异常捕获
        #捕获 httpx 请求相关的错误
        logging.error(f"获取服务器状态时发生请求错误: {str(e)}")
    except Exception as e:
        # 捕获其他潜在错误
        logging.error(f"获取服务器状态时发生未知错误: {str(e)}")
    
    return None

async def print_server_status():
    """打印服务器状态"""
    status_data = await get_server_status()
    
    if not status_data:
        print("无法获取服务器状态")
        return
    
    print("\n" + "="*50)
    print("服务器LLM池状态:")
    print("="*50)
    
    # 大模型池状态
    large_pool = status_data.get("llm_pool_status", {}).get("large_pool", {})
    small_pool = status_data.get("llm_pool_status", {}).get("small_pool", {})
    
    # 显示大模型池
    print("\n大模型池:")
    large_table = prettytable.PrettyTable()
    large_table.field_names = ["模型", "URL", "并发", "最大并发", "总请求", "可用"]
    
    for instance in large_pool.get("instances", []):
        large_table.add_row([
            instance.get("model", "未知"),
            instance.get("url", "未知"),
            instance.get("current_concurrency", 0),
            instance.get("max_concurrency", 0),
            instance.get("total_requests", 0),
            "✓" if instance.get("available", False) else "✗"
        ])
    
    print(large_table)
    print(f"利用率: {large_pool.get('utilization_percent', 0)}%, 可用实例: {large_pool.get('available_instance_count', 0)}/{large_pool.get('instance_count', 0)}")
    
    # 显示小模型池
    print("\n小模型池:")
    small_table = prettytable.PrettyTable()
    small_table.field_names = ["模型", "URL", "并发", "最大并发", "总请求", "可用"]
    
    for instance in small_pool.get("instances", []):
        small_table.add_row([
            instance.get("model", "未知"),
            instance.get("url", "未知"),
            instance.get("current_concurrency", 0),
            instance.get("max_concurrency", 0),
            instance.get("total_requests", 0),
            "✓" if instance.get("available", False) else "✗"
        ])
    
    print(small_table)
    print(f"利用率: {small_pool.get('utilization_percent', 0)}%, 可用实例: {small_pool.get('available_instance_count', 0)}/{small_pool.get('instance_count', 0)}")
    
    # 总体统计
    overall = status_data.get("llm_pool_status", {}).get("overall", {})
    print("\n总体状态:")
    print(f"总实例数: {overall.get('total_instances', 0)}, 可用实例: {overall.get('available_instances', 0)}")
    print(f"总容量: {overall.get('total_capacity', 0)}, 当前负载: {overall.get('current_load', 0)}")
    print(f"总体利用率: {overall.get('overall_utilization_percent', 0)}%")
    print("="*50)

async def run_concurrent_test(concurrent_requests: int, model_type: str = None):
    """运行并发测试"""
    print(f"开始LLM池并发测试 - 发送{concurrent_requests}个并发请求")
    
    # 创建统计对象
    stats = RequestStats()
    
    # 打印服务器初始状态
    await print_server_status()
    
    # 创建请求任务
    tasks = []

    # 分配模型类型
    models = []
    if model_type:
        # 指定了模型类型，全部使用该类型
        models = [model_type] * concurrent_requests
    else:
        # 未指定模型类型，随机分配大模型和小模型
        models = []
        for i in range(concurrent_requests):
            # 按5:3:2的比例分配large:small:default
            rand = random.random()
            if rand < 0.5:
                models.append("large")
            elif rand < 0.8:
                models.append("small")
            else:
                models.append("default")
    
    for i in range(concurrent_requests):
        request_id = str(uuid.uuid4())
        prompt = random.choice(TEST_PROMPTS)
        # 确保并发发送，不要等待前一个请求完成
        task = send_request(request_id, prompt, models[i], stats)
        tasks.append(asyncio.create_task(task))
    
    # 创建统计打印任务
    stats_task = asyncio.create_task(print_live_stats(stats, concurrent_requests))
    
    # 等待所有请求完成
    await asyncio.gather(*tasks)
    
    # 等待统计任务完成
    if not stats_task.done():
        stats_task.cancel()
        try:
            await stats_task
        except asyncio.CancelledError:
            pass
    
    # 打印最终统计信息
    await print_final_stats(stats)
    
    # 打印服务器最终状态
    await print_server_status()

async def main():
    """主函数"""
    # 声明全局变量
    global SERVER_URL, CONCURRENT_REQUESTS
    
    parser = argparse.ArgumentParser(description='LLM池系统并发测试')
    parser.add_argument('-c', '--concurrent', type=int, default=CONCURRENT_REQUESTS,
                        help=f'并发请求数 (默认: {CONCURRENT_REQUESTS})')
    parser.add_argument('-m', '--model', type=str, choices=['large', 'small', 'default'], 
                        help='指定使用的模型类型 (默认: 随机分配)')
    parser.add_argument('-s', '--server', type=str, default=SERVER_URL,
                        help=f'服务器地址 (默认: {SERVER_URL})')
    
    args = parser.parse_args()
    
    # 更新全局配置
    SERVER_URL = args.server
    
    # 运行测试
    await run_concurrent_test(args.concurrent, args.model)

if __name__ == "__main__":
    asyncio.run(main()) 