# RAG嵌入模型问题解决方案

## 问题描述

在使用 `Chroma.from_documents` 创建向量数据库时出现以下错误：

```
AttributeError: 'NoneType' object has no attribute 'get'
```

这个错误是由于嵌入模型的接口实现不正确导致的。具体来说，当尝试使用 `embedding = embedding_cls.get_embedding_fun()` 这种方式传递嵌入函数时，可能遇到返回值为 `None` 的情况。

## 解决方案

我们从根本上重写了 `RagEmbedding` 类，主要做了以下改进：

1. **正确实现 `Embeddings` 接口**：
   - 让 `RagEmbedding` 类直接继承 `Embeddings` 接口
   - 实现必要的 `embed_documents` 和 `embed_query` 方法

2. **使用最新的 HuggingFace 嵌入包**：
   - 从 `langchain_huggingface` 导入 `HuggingFaceEmbeddings`
   - 这解决了与旧版 LangChain 的兼容性问题

3. **简化 Chroma 数据库创建**：
   - 直接将 `embedding_cls` 实例传递给 `Chroma.from_documents` 的 `embedding` 参数
   - 不再调用 `get_embedding_fun()` 方法

## 代码修改

新的 `RagEmbedding` 类实现：

```python
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

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
```

正确的 Chroma 数据库创建方式：

```python
from model import RagEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 初始化嵌入模型
embedding_cls = RagEmbedding(model_name="BAAI/bge-m3")

# 创建文档
documents = [
    Document(page_content="文档内容", metadata={"source": "来源"}),
    # 更多文档...
]

# 创建 Chroma 数据库
embedding_db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_cls,  # 直接传递嵌入类实例
    persist_directory="./chroma_db",
    collection_name="zhidu_db"
)
```

重新加载数据库的正确方式：

```python
# 重新加载数据库
loaded_db = Chroma(
    collection_name="zhidu_db",
    embedding_function=embedding_cls,  # 注意参数名是embedding_function
    persist_directory="./chroma_db"
)
```

## 依赖安装

确保安装了所有必要的依赖：

```bash
pip install langchain langchain-huggingface langchain-chroma chromadb transformers torch
```

## 完整示例

请查看 `solution.py` 文件，这是一个完整的工作示例，展示了如何创建、使用和重新加载 Chroma 向量数据库。

```bash
python solution.py
```

这个示例包含：
1. 创建示例文档
2. 初始化嵌入模型
3. 创建 Chroma 数据库
4. 执行相似性搜索
5. 重新加载数据库并验证

## 结论

通过正确实现 `Embeddings` 接口并使用最新的 LangChain 和 HuggingFace 包，我们解决了 `NoneType` 对象没有 `get` 属性的错误。这个解决方案不仅修复了错误，还提供了一个更简洁、更可靠的嵌入模型实现。 