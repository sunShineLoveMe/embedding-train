# RAG 嵌入模型训练项目

本项目实现了一个用于 RAG (检索增强生成) 系统的嵌入模型，使用 BAAI/bge-m3 作为底层模型。

## 项目背景

RAG 系统需要将文本转换为向量表示（嵌入），以便进行相似性搜索和检索。本项目使用 Hugging Face 的预训练模型 BAAI/bge-m3 来生成这些嵌入，并使用 Chroma 向量数据库来存储和检索。

## 项目结构

- `model.py`: 定义了 RAG 系统的两个核心组件：
  - `RagLLM`: 连接 Qwen2.5-72B 大语言模型
  - `RagEmbedding`: 使用 HuggingFaceEmbeddings 封装的嵌入模型
- `solution.py`: 展示如何创建和使用 Chroma 向量数据库的示例代码
- `test_chroma.py`: 测试 Chroma 数据库创建和查询的简单测试脚本
- `test_embedding.py`: 测试嵌入模型基本功能的测试脚本
- `local_chroma_solution.py`: 使用本地文件系统模式创建和使用Chroma数据库的示例代码
- `rag_llm_local.ipynb`: 使用本地文件系统模式的Jupyter Notebook版本

## 安装依赖

```bash
pip install langchain langchain-huggingface langchain-chroma chromadb transformers torch
```

## 使用方法

### 方式一：本地文件系统模式（推荐）

1. 初始化嵌入模型：

```python
from model import RagEmbedding

# 初始化嵌入模型
embedding_cls = RagEmbedding(model_name="BAAI/bge-m3", device="cpu")  # 或 device="cuda" 使用 GPU
```

2. 创建向量数据库：

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 创建文档
documents = [
    Document(page_content="文档内容1", metadata={"source": "来源1"}),
    Document(page_content="文档内容2", metadata={"source": "来源2"}),
]

# 创建 Chroma 数据库（使用本地文件系统模式）
db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_cls,
    persist_directory="./chroma_db",  # 指定保存目录
    collection_name="my_collection"
)

# 持久化数据库 (注意：最新版本的Chroma不需要手动调用persist()方法)
# db.persist()  # 旧版本需要这一行
```

3. 使用向量数据库进行相似性搜索：

```python
# 相似性搜索
results = db.similarity_search("查询文本", k=3)  # 返回最相似的3个文档
for doc in results:
    print(doc.page_content, doc.metadata)
```

4. 重新加载数据库：

```python
# 重新加载已存在的数据库
loaded_db = Chroma(
    collection_name="my_collection",
    embedding_function=embedding_cls,
    persist_directory="./chroma_db"
)
```

### 方式二：服务器模式

1. 启动 Chroma 服务器（需要单独安装）：

```bash
# 安装chromadb服务器
pip install chromadb

# 启动服务器
chroma run --path ./chroma_server_data --host localhost --port 8000
```

2. 连接到 Chroma 服务器：

```python
import chromadb
from langchain_chroma import Chroma

# 连接到Chroma服务器
chroma_client = chromadb.HttpClient(host="localhost", port="8000")

# 使用客户端创建数据库
db = Chroma.from_documents(
    documents=documents,
    embedding=embedding_cls,
    client=chroma_client,
    collection_name="my_collection"
)
```

## 常见问题解决

1. **Connection reset by peer 错误**:
   - 这个错误通常是因为无法连接到Chroma服务器导致的
   - 解决方案：
     - 确保Chroma服务器已启动并可访问
     - 或者改用本地文件系统模式（推荐）：使用`persist_directory`参数而不是`client`参数
     - 可以运行`local_chroma_solution.py`或`rag_llm_local.ipynb`查看示例

2. **AttributeError: 'NoneType' object has no attribute 'get'**:
   - 这个错误通常是因为嵌入模型接口不正确造成的
   - 确保 `RagEmbedding` 类正确实现了 `Embeddings` 接口，包含 `embed_documents` 和 `embed_query` 方法
   - 使用 `Chroma.from_documents` 时，直接传递 `embedding_cls` 实例，而不是调用 `get_embedding_fun()`

3. **Chroma 客户端创建失败**:
   - 最新版本的 Chroma 不再使用 `client` 参数
   - 使用 `persist_directory` 参数来指定存储位置

4. **TypeError: Chroma.__init__() got an unexpected keyword argument 'embedding'**:
   - 更新后的 Chroma 使用 `embedding_function` 参数而不是 `embedding`
   - 重新加载时使用 `embedding_function=embedding_cls`

## 例子

1. 使用本地文件系统模式：

```bash
python local_chroma_solution.py
```

2. 使用服务器模式（需要先启动服务器）：

```bash
python solution.py
```

## 注意事项

1. 第一次使用时会自动下载模型（约 1.5GB），这可能需要一些时间
2. 默认使用 CPU 进行计算，如果有 GPU 可以设置 `device="cuda"` 以提高性能
3. 对于大量文档，建议使用批处理方式进行嵌入计算
4. 确保安装了所有必要的依赖
5. 推荐使用本地文件系统模式，它更简单且不依赖外部服务 