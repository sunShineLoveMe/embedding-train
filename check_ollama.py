#!/usr/bin/env python

"""
Ollama连接状态检查工具
可以帮助排查Ollama连接问题，检查服务是否正常运行，模型是否已加载
"""

import requests
import sys
import time
import json
import os
import subprocess
import platform

def red(text):
    return f"\033[91m{text}\033[0m"

def green(text):
    return f"\033[92m{text}\033[0m"

def yellow(text):
    return f"\033[93m{text}\033[0m"

def blue(text):
    return f"\033[94m{text}\033[0m"

def check_ollama_running():
    """检查Ollama服务是否运行"""
    try:
        # 检查服务是否运行
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            return True, response.json()
        return False, f"服务返回状态码: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "无法连接到Ollama服务，服务可能未运行"
    except requests.exceptions.Timeout:
        return False, "连接Ollama服务超时"
    except Exception as e:
        return False, f"检查服务时发生错误: {str(e)}"

def check_model_available(model_name):
    """检查指定模型是否已加载到Ollama中"""
    try:
        response = requests.post(
            "http://localhost:11434/api/show",
            json={"name": model_name},
            timeout=5
        )
        
        if response.status_code == 200:
            model_info = response.json()
            return True, model_info
        return False, f"模型检查返回状态码: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "无法连接到Ollama服务，服务可能未运行"
    except requests.exceptions.Timeout:
        return False, "连接Ollama服务超时"
    except Exception as e:
        return False, f"检查模型时发生错误: {str(e)}"

def test_model_response(model_name):
    """测试模型响应速度"""
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "你好，这是一个测试请求",
                "stream": False
            },
            timeout=20
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return True, (result.get("response", ""), elapsed)
        return False, f"测试模型返回状态码: {response.status_code}, 耗时: {elapsed:.2f}秒"
    except requests.exceptions.ConnectionError:
        return False, "无法连接到Ollama服务，服务可能未运行"
    except requests.exceptions.Timeout:
        return False, "测试模型响应超时"
    except Exception as e:
        return False, f"测试模型时发生错误: {str(e)}"

def check_system_resources():
    """检查系统资源使用情况"""
    try:
        # 查找Ollama进程
        ollama_processes = []
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            output = subprocess.check_output(["ps", "aux"], universal_newlines=True)
            for line in output.splitlines():
                if "ollama" in line:
                    ollama_processes.append(line)
        elif system == 'Linux':
            output = subprocess.check_output(["ps", "aux"], universal_newlines=True)
            for line in output.splitlines():
                if "ollama" in line:
                    ollama_processes.append(line)
        elif system == 'Windows':
            output = subprocess.check_output(["tasklist", "/fi", "imagename eq ollama*"], universal_newlines=True)
            ollama_processes = [line for line in output.splitlines() if "ollama" in line.lower()]
        
        return True, ollama_processes
    except Exception as e:
        return False, f"检查系统资源时发生错误: {str(e)}"

def kill_duplicate_instances():
    """尝试终止多余的Ollama实例"""
    try:
        system = platform.system()
        
        if system == 'Darwin' or system == 'Linux':
            # 这里仅终止重复的ollama run命令，保留服务
            subprocess.run(["pkill", "-f", "ollama run"])
            return True, "已尝试关闭多余的ollama run实例"
        elif system == 'Windows':
            # Windows系统需要更复杂的处理
            return False, "Windows系统请手动关闭多余的Ollama实例"
        
        return False, "未知系统类型，请手动关闭多余的Ollama实例"
    except Exception as e:
        return False, f"终止多余实例时发生错误: {str(e)}"

def main():
    model_name = "qwen:14b"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    print(blue(f"==== Ollama 状态检查工具 ===="))
    print(f"检查模型: {model_name}")
    print()
    
    # 检查Ollama服务
    print(blue("1. 检查Ollama服务状态..."))
    success, result = check_ollama_running()
    if success:
        print(green("✓ Ollama服务运行正常"))
        if isinstance(result, dict) and 'models' in result:
            print(f"   已加载的模型: {', '.join([m['name'] for m in result['models']])}")
    else:
        print(red(f"✗ Ollama服务异常: {result}"))
        print(yellow("建议: 请确认Ollama应用是否已启动"))
        sys.exit(1)
    
    # 检查指定模型
    print()
    print(blue(f"2. 检查模型 {model_name} 状态..."))
    success, result = check_model_available(model_name)
    if success:
        print(green(f"✓ 模型 {model_name} 已加载"))
        if isinstance(result, dict):
            if 'parameters' in result:
                print(f"   参数: {json.dumps(result['parameters'], indent=2)}")
    else:
        print(red(f"✗ 模型状态异常: {result}"))
        print(yellow(f"建议: 请运行 'ollama pull {model_name}' 拉取模型"))
        sys.exit(1)
    
    # 测试模型响应
    print()
    print(blue("3. 测试模型响应..."))
    success, result = test_model_response(model_name)
    if success:
        response, elapsed = result
        print(green(f"✓ 模型响应正常，耗时: {elapsed:.2f}秒"))
        print(f"   回复: {response[:100]}..." if len(response) > 100 else f"   回复: {response}")
    else:
        print(red(f"✗ 模型响应异常: {result}"))
        print(yellow("建议: 模型可能过大或正在加载中，请稍后再试"))
    
    # 检查系统资源
    print()
    print(blue("4. 检查Ollama进程..."))
    success, result = check_system_resources()
    if success:
        if len(result) > 0:
            ollama_process_count = len(result)
            print(f"发现 {ollama_process_count} 个Ollama相关进程:")
            for i, proc in enumerate(result):
                print(f"   {i+1}. {proc}")
            
            if ollama_process_count > 2:
                print(yellow("! 警告: 发现多个Ollama进程，可能导致资源冲突"))
                print(yellow("建议: 保留一个Ollama服务实例，关闭其他实例"))
                
                choice = input("是否尝试关闭多余的Ollama实例? (y/n): ")
                if choice.lower() == 'y':
                    success, msg = kill_duplicate_instances()
                    if success:
                        print(green(f"✓ {msg}"))
                    else:
                        print(yellow(f"! {msg}"))
        else:
            print(yellow("! 未发现Ollama进程，服务可能由其他方式启动"))
    else:
        print(red(f"✗ 检查进程异常: {result}"))
    
    print()
    print(blue("==== 检查完成 ===="))
    print("如需使用指定模型: python check_ollama.py <model_name>")

if __name__ == "__main__":
    main() 