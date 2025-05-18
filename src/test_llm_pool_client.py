#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import time
import uuid
import json
import logging
import random
from typing import Dict, Any, List
import httpx
from tqdm import tqdm
import prettytable
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 测试配置
SERVER_URL = "http://localhost:8000"  # 服务器地址
API_KEY = "test-api-key"  # API密钥
CONCURRENT_REQUESTS = 20  # 并发请求数
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

async def send_request(request_id: str, prompt: str, stats: RequestStats) -> None:
    """
    发送请求到LLM池服务器
    
    Args:
        request_id: 请求ID
        prompt: 提示词
        stats: 统计对象
    """
    try:
        # 记录请求开始
        await stats.start_request(request_id)
        
        # 构造请求数据
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "pool_type": "default",  # 使用默认池（大模型池）
            "request_id": request_id,
            "stream": True,
            "model": "default"  # 添加model字段，设为default
        }
        
        # 发送请求
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{SERVER_URL}/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                
                # 解析响应
                content = ""
                tokens = 0
                llm_name = "未知LLM"
                
                if "usage" in response_data:
                    tokens = response_data["usage"].get("total_tokens", 0)
                
                if "llm_info" in response_data:
                    llm_name = response_data["llm_info"]
                
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    if "message" in response_data["choices"][0]:
                        content = response_data["choices"][0]["message"].get("content", "")
                
                # 记录请求完成
                await stats.complete_request(request_id, llm_name, response_time, tokens)
                
                logging.info(f"[{request_id}] 请求完成: {llm_name}, 耗时: {response_time:.2f}秒, Tokens: {tokens}")
                
                # 仅打印内容的前30个字符
                content_preview = content[:30] + "..." if len(content) > 30 else content
                logging.debug(f"[{request_id}] 响应内容: {content_preview}")
            else:
                error = f"HTTP错误: {response.status_code}, {response.text}"
                await stats.fail_request(request_id, error)
                logging.error(f"[{request_id}] {error}")
    
    except Exception as e:
        error = f"请求异常: {str(e)}"
        await stats.fail_request(request_id, error)
        logging.error(f"[{request_id}] {error}")

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
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{SERVER_URL}/status")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logging.error(f"获取服务器状态失败: {str(e)}")
    
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

async def main():
    """主函数"""
    print(f"开始LLM池并发测试 - 发送{CONCURRENT_REQUESTS}个并发请求")
    
    # 创建统计对象
    stats = RequestStats()
    
    # 打印服务器初始状态
    await print_server_status()
    
    # 创建请求任务
    tasks = []
    for i in range(CONCURRENT_REQUESTS):
        request_id = str(uuid.uuid4())
        prompt = random.choice(TEST_PROMPTS)
        task = asyncio.create_task(send_request(request_id, prompt, stats))
        tasks.append(task)
    
    # 创建统计打印任务
    stats_task = asyncio.create_task(print_live_stats(stats, CONCURRENT_REQUESTS))
    
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

if __name__ == "__main__":
    asyncio.run(main()) 