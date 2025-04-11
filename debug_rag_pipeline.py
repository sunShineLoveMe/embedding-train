"""
RAG管道调试脚本
用于测试和诊断run_rag_pipline函数的执行情况
"""

import requests
import json
from model import QwenLLM, safe_json_parse
import time

def test_raw_api_call(query, context):
    """直接测试Ollama API调用"""
    url = "http://localhost:11434/api/generate"
    
    # 准备提示词
    prompt = f"""根据以下上下文，回答用户的问题。
    
上下文:
{context}

用户问题: {query}

请只回答上下文中包含的信息，如果上下文中没有相关信息，请说明"抱歉，我无法根据提供的上下文回答这个问题。"
"""
    
    payload = {
        "model": "qwen:14b",
        "prompt": prompt,
        "stream": False,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 2048
    }
    
    try:
        print("发送请求到Ollama API...")
        print("-" * 80)
        print(f"请求内容(前200字符):\n{prompt[:200]}...")
        print("-" * 80)
        
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        elapsed = time.time() - start_time
        
        print(f"请求完成，状态码: {response.status_code}，耗时: {elapsed:.2f}秒")
        print(f"响应头: {dict(response.headers)}")
        
        # 打印原始响应
        print("-" * 80)
        print(f"原始响应内容(前500字符):\n{response.text[:500]}...")
        print("-" * 80)
        
        # 尝试解析JSON
        try:
            result = safe_json_parse(response.content)
            print("JSON解析成功:")
            print(f"结果: {result}")
        except Exception as e:
            print(f"JSON解析失败: {str(e)}")
        
        return response.text
    except Exception as e:
        print(f"请求错误: {str(e)}")
        return None

def debug_rag_llm(query, context):
    """使用QwenLLM类进行调试"""
    llm = QwenLLM(timeout=60, max_retries=2)
    
    # 准备提示词
    prompt = f"""根据以下上下文，回答用户的问题。
    
上下文:
{context}

用户问题: {query}

请只回答上下文中包含的信息，如果上下文中没有相关信息，请说明"抱歉，我无法根据提供的上下文回答这个问题。"
"""
    
    print("使用QwenLLM类进行调用...")
    print("-" * 80)
    
    start_time = time.time()
    result = llm._call(prompt, verbose=True)
    elapsed = time.time() - start_time
    
    print(f"调用完成，耗时: {elapsed:.2f}秒")
    print("-" * 80)
    print(f"结果: {result}")
    
    return result

def simulate_run_rag_pipeline():
    """模拟run_rag_pipline函数的执行"""
    # 示例查询和上下文
    query = "那个，我们公司有什么规定来着？"
    context = """上下文1: 1、学校的工作时间由学校决定并公布。学校内除特殊岗位特别规定外，全体教职员工均应严格执行学校的作息时间表，不迟到、不早退、不中途离校 

上下文2: 1、学校的工作时间由学校决定并公布。学校内除特殊岗位特别规定外，全体教职员工均应严格执行学校的作息时间表，不迟到、不早退、不中途离校 

上下文3: 企业形象是学校非常重要的财富，维护好企业形象是每个员工必须遵守的规则。员工必须严格遵守学校的企业文化，经营理念和管理制度，充分维护和支持学校的企业形象建设。任何人不得出现有损学校团队建设、诋毁学校企业管理和企业文化的语言行为"""
    
    print("=" * 80)
    print("RAG管道调试")
    print("=" * 80)
    print(f"查询: {query}")
    print(f"上下文: {context}")
    print("=" * 80)
    
    # 方法1: 直接API调用
    print("\n[方法1] 直接API调用")
    raw_result = test_raw_api_call(query, context)
    
    # 方法2: 使用QwenLLM类
    print("\n[方法2] 使用QwenLLM类")
    llm_result = debug_rag_llm(query, context)
    
    print("\n调试完成。")

if __name__ == "__main__":
    simulate_run_rag_pipeline() 