import warnings
warnings.filterwarnings('ignore')
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict, ClassVar
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import chromadb
import langchain
from transformers import AutoModel, AutoTokenizer
import torch
import requests
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 这个文件实现了RAG系统的两个核心组件
# 大语言模型的接口：链接本地Ollama的Qwen:14b模型
# 嵌入模型的接口：使用硅基流动平台的BAAI/bge-m3模型进行文本转化为向量，用于文档检索

print("ChromaDB version:", chromadb.__version__)
print("Langchain version:", langchain.__version__)

class RagLLM(object):
    def __init__(self, model_name="qwen:14b", api_base="http://localhost:11434/api", timeout=10):
        super().__init__()
        self.model_name = model_name
        self.api_base = api_base
        self.timeout = timeout
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError))
    )
    def __call__(self, prompt: str, **kwargs: Any):
        url = f"{self.api_base}/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": kwargs.get('stream', False),
            "temperature": kwargs.get('temperature', 0.1),
            "top_p": kwargs.get('top_p', 0.9),
            "max_tokens": kwargs.get('max_tokens', 4096)
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()  # 确保请求成功
            result = response.json()
            
            # 打印请求耗时，帮助调试
            elapsed = time.time() - start_time
            if kwargs.get('verbose', False):
                print(f"Ollama API request completed in {elapsed:.2f}s")
            
            if kwargs.get('stream', False):
                return result
            
            return result.get('response', '')
        except requests.exceptions.Timeout:
            print(f"Error: API request timed out after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return ""


class QwenLLM(LLM):
    # 定义Pydantic字段
    model_name: str = "qwen:14b"
    api_base: str = "http://localhost:11434/api"
    timeout: int = 30
    max_retries: int = 3
    
    def __init__(self, 
                 model_name="qwen:14b", 
                 api_base="http://localhost:11434/api", 
                 timeout=30,
                 max_retries=3,
                 **kwargs):
        # 初始化父类
        super().__init__(**kwargs)
        # 设置实例属性
        self.model_name = model_name
        self.api_base = api_base
        self.timeout = timeout
        self.max_retries = max_retries

    def _call(self, 
              prompt: str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        
        # 重试装饰器无法直接用于类方法，所以在方法内部实现重试逻辑
        attempts = 0
        last_exception = None
        backoff = 1
        
        while attempts < self.max_retries:
            try:
                # 确保prompt是字符串类型
                if not isinstance(prompt, str):
                    prompt = str(prompt)
                
                url = f"{self.api_base}/generate"
                
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,  # 对于评估，我们不使用流式输出
                    "temperature": kwargs.get('temperature', 0.1),
                    "top_p": kwargs.get('top_p', 0.9),
                    "max_tokens": kwargs.get('max_tokens', 4096)
                }
                
                # 如果提供了stop参数，添加到请求中
                if stop:
                    payload["stop"] = stop
                
                start_time = time.time()
                if run_manager:
                    run_manager.on_text("向Ollama发送请求...\n", verbose=kwargs.get("verbose", False))
                
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()  # 确保请求成功
                result = response.json()
                
                # 记录API调用统计
                elapsed = time.time() - start_time
                if kwargs.get('verbose', False):
                    print(f"API request completed in {elapsed:.2f}s")
                
                if run_manager:
                    run_manager.on_text(f"请求完成，耗时: {elapsed:.2f}秒\n", verbose=kwargs.get("verbose", False))
                
                # 获取生成的文本
                generated_text = result.get('response', '')
                
                # 确保返回的是非空字符串
                if not generated_text or not isinstance(generated_text, str):
                    if run_manager:
                        run_manager.on_text("警告: 服务器返回空响应\n", verbose=True)
                    return ""
                    
                # 清理和格式化响应文本
                generated_text = generated_text.strip()
                
                return generated_text
                
            except requests.exceptions.Timeout:
                attempts += 1
                last_exception = f"请求超时（第{attempts}次尝试）"
                if run_manager:
                    run_manager.on_text(f"警告: {last_exception}\n", verbose=True)
                print(f"警告: API请求超时，第{attempts}/{self.max_retries}次尝试")
                
                # 指数退避
                if attempts < self.max_retries:
                    sleep_time = backoff
                    backoff *= 2
                    print(f"等待{sleep_time}秒后重试...")
                    time.sleep(sleep_time)
                
            except requests.exceptions.RequestException as e:
                attempts += 1
                last_exception = f"请求错误: {str(e)}（第{attempts}次尝试）"
                if run_manager:
                    run_manager.on_text(f"警告: {last_exception}\n", verbose=True)
                print(f"警告: API请求错误: {str(e)}，第{attempts}/{self.max_retries}次尝试")
                
                # 指数退避
                if attempts < self.max_retries:
                    sleep_time = backoff
                    backoff *= 2
                    print(f"等待{sleep_time}秒后重试...")
                    time.sleep(sleep_time)
                
            except Exception as e:
                print(f"未预期错误: {str(e)}")
                if run_manager:
                    run_manager.on_text(f"错误: {str(e)}\n", verbose=True)
                return ""  # 发生未预期错误时返回空字符串
        
        # 如果所有尝试都失败
        print(f"错误: 所有{self.max_retries}次尝试均失败: {last_exception}")
        if run_manager:
            run_manager.on_text(f"错误: 所有{self.max_retries}次尝试均失败\n", verbose=True)
        return ""
    
    @property
    def _llm_type(self) -> str:
        return "ollama_qwen_llm"
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name, 
            "api_base": self.api_base,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
        
    # 一个实用方法，用于测试API连接
    def test_connection(self):
        """测试与Ollama API的连接"""
        try:
            url = f"{self.api_base.rstrip('/api')}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return True, "连接成功"
        except Exception as e:
            return False, f"连接失败: {str(e)}"

class RagEmbedding(Embeddings):
    def __init__(self, model_name="BAAI/bge-m3", device="cpu"):
        """Initialize the RagEmbedding class using HuggingFaceEmbeddings."""
        super().__init__()
        try:
            self.embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device}
            )
            print(f"模型加载成功，使用设备: {device}")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents."""
        try:
            return self.embedding.embed_documents(texts)
        except Exception as e:
            print(f"文档嵌入失败: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        try:
            return self.embedding.embed_query(text)
        except Exception as e:
            print(f"查询嵌入失败: {str(e)}")
            raise