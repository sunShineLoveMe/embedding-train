import warnings
warnings.filterwarnings('ignore')
from langchain.llms.base import LLM
from typing import Any, List, Optional
from openai import OpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import chromadb
import langchain
from transformers import AutoModel, AutoTokenizer
import torch

# 这个文件实现了RAG系统的两个核心组件
# 大语言模型的接口：链接硅基流动平台的Qwen2-72B模型
# 嵌入模型的接口：使用硅基流动平台的BAAI/bge-m3模型进行文本转化为向量，用于文档检索

print("ChromaDB version:", chromadb.__version__)
print("Langchain version:", langchain.__version__)

class RagLLM(object):
    client: Optional[Any] = None
    def __init__(self):
        super().__init__()
        self.client = OpenAI(base_url="https://api.siliconflow.cn/v1",
                             api_key="sk-yexkbpbyfcingpdzpiiaiokqbvdlzroatlkqxyqfnpsebaen")

        
    def __call__(self, prompt : str, **kwargs: Any):
        completion = self.client.completions.create(model="Qwen/Qwen2.5-72B-Instruct", 
                                                    prompt=prompt,
                                                    temperature=kwargs.get('temperature', 0.1),
                                                    top_p=kwargs.get('top_p', 0.9),
                                                    max_tokens=kwargs.get('max_tokens', 4096), 
                                                    stream=kwargs.get('stream', False))
        if kwargs.get('stream', False):
            return completion
        return completion.choices[0].text
    


class QwenLLM(LLM):
    client: Optional[Any] = None
    def __init__(self):
        super().__init__()
        self.client = OpenAI(base_url="https://api.siliconflow.cn/v1",
                             api_key="sk-yexkbpbyfcingpdzpiiaiokqbvdlzroatlkqxyqfnpsebaen")

    def _call(self, 
              prompt: str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        try:
            # 确保prompt是字符串类型
            if not isinstance(prompt, str):
                prompt = str(prompt)
            
            # 调用API获取响应
            completion = self.client.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct", 
                prompt=prompt,
                temperature=kwargs.get('temperature', 0.1),
                top_p=kwargs.get('top_p', 0.9),
                max_tokens=kwargs.get('max_tokens', 4096), 
                stream=False  # 对于评估，我们不使用流式输出
            )
            
            # 确保返回有效的文本响应
            if not completion.choices:
                return ""
            
            response = completion.choices[0].text
            
            # 确保返回的是非空字符串
            if not response or not isinstance(response, str):
                return ""
                
            # 清理和格式化响应文本
            response = response.strip()
            
            return response
            
        except Exception as e:
            print(f"Error in QwenLLM._call: {str(e)}")
            return ""  # 在出错时返回空字符串而不是抛出异常    
    # def _call(self, 
    #           prompt : str, 
    #           stop: Optional[List[str]] = None,
    #           run_manager: Optional[CallbackManagerForLLMRun] = None,
    #           **kwargs: Any):
    #     completion = self.client.completions.create(model="Qwen/Qwen2.5-72B-Instruct", 
    #                                                 prompt=prompt,
    #                                                 temperature=kwargs.get('temperature', 0.1),
    #                                                 top_p=kwargs.get('top_p', 0.9),
    #                                                 max_tokens=kwargs.get('max_tokens', 4096), 
    #                                                 stream=kwargs.get('stream', False))
    #     return completion.choices[0].text
    
    @property
    def _llm_type(self) -> str:
        return "rag_llm_qwen2.5_72b"  

# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# class RagEmbedding(object):
#     def __init__(self, model_path="./data/llm_app/embedding_models/bge-m3//", 
#                  device="cpu"):
#         self.embedding = HuggingFaceEmbeddings(model_name=model_path,
#                                                model_kwargs={"device": "cpu"})
#     def get_embedding_fun(self):
#         return self.embedding

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