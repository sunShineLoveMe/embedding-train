"""
在Jupyter Notebook中使用的解决方案
直接使用ChromaDB API避开langchain_chroma层可能存在的问题
将此代码复制到Jupyter Notebook中运行
"""

# 步骤1: 导入必要的库
import chromadb
from model import RagEmbedding
from langchain_core.documents import Document
import numpy as np
import uuid

# 步骤2: 配置Chroma服务器信息
CHROMA_SERVER_HOST = "localhost"
CHROMA_SERVER_PORT = "8000"  # 默认端口，如果你修改过，请使用实际端口
COLLECTION_NAME = "notebook_db"  # 可以修改为你想要的集合名称

# 步骤3: 创建ChromaDB客户端
client = chromadb.HttpClient(
    host=CHROMA_SERVER_HOST,
    port=CHROMA_SERVER_PORT
)
print(f"成功连接到Chroma服务器: http://{CHROMA_SERVER_HOST}:{CHROMA_SERVER_PORT}")

# 步骤4: 检查并清理同名集合
existing_collections = client.list_collections()
for collection in existing_collections:
    if collection.name == COLLECTION_NAME:
        client.delete_collection(COLLECTION_NAME)
        print(f"已删除现有的集合: {COLLECTION_NAME}")
        break

# 步骤5: 创建新的集合
collection = client.create_collection(name=COLLECTION_NAME)
print(f"成功创建集合: {COLLECTION_NAME}")

# 步骤6: 初始化嵌入模型
embedding_model = RagEmbedding(model_name="BAAI/bge-m3", device="cpu")

# 步骤7: 从现有文档列表准备数据
def prepare_documents_for_chroma(documents):
    """将Document对象列表准备为Chroma格式的数据"""
    texts = [doc.page_content for doc in documents]
    ids = [f"doc_{uuid.uuid4()}" for _ in range(len(documents))]
    metadatas = [doc.metadata for doc in documents]
    
    # 使用嵌入模型生成向量
    embeddings = embedding_model.embed_documents(texts)
    
    return {
        "texts": texts,
        "ids": ids, 
        "metadatas": metadatas,
        "embeddings": embeddings
    }

# 步骤8: 添加示例文档
example_documents = [
    Document(page_content="中国是一个有着悠久历史的国家", metadata={"source": "中国简介"}),
    Document(page_content="北京是中国的首都，是政治和文化中心", metadata={"source": "北京简介"}),
    Document(page_content="上海是中国最大的经济中心城市", metadata={"source": "上海简介"}),
    Document(page_content="广州是中国南方重要的商业城市", metadata={"source": "广州简介"}),
    Document(page_content="深圳是中国重要的科技创新中心", metadata={"source": "深圳简介"}),
]

# 添加文档到数据库
prepared_data = prepare_documents_for_chroma(example_documents)
collection.add(
    documents=prepared_data["texts"],
    embeddings=prepared_data["embeddings"],
    ids=prepared_data["ids"],
    metadatas=prepared_data["metadatas"]
)
print(f"成功添加 {len(example_documents)} 个示例文档到集合")

# 步骤9: 定义相似性搜索函数
def similarity_search(query, k=3):
    """执行相似性搜索并返回结果"""
    # 获取查询的嵌入向量
    query_embedding = embedding_model.embed_query(query)
    
    # 执行相似性搜索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # 将结果转换为可读格式
    documents = []
    if results['documents'] and len(results['documents']) > 0:
        for i, doc_text in enumerate(results['documents'][0]):
            if i < len(results['metadatas'][0]):
                metadata = results['metadatas'][0][i]
                documents.append(Document(page_content=doc_text, metadata=metadata))
    
    return documents

# 步骤10: 测试相似性搜索
query = "中国的城市"
results = similarity_search(query, k=3)
print(f"\n查询: '{query}'")
print("相似度最高的3个文档:")
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content} (来源: {doc.metadata['source']})")

# 步骤11: 添加到RAG系统的处理函数（使用示例）
def add_documents_to_chroma(documents, collection_name=COLLECTION_NAME):
    """将新文档添加到现有集合中"""
    # 获取集合
    collection = client.get_collection(name=collection_name)
    
    # 准备数据
    prepared_data = prepare_documents_for_chroma(documents)
    
    # 添加到集合
    collection.add(
        documents=prepared_data["texts"],
        embeddings=prepared_data["embeddings"],
        ids=prepared_data["ids"],
        metadatas=prepared_data["metadatas"]
    )
    
    return f"成功添加 {len(documents)} 个文档到集合 {collection_name}"

def query_similar_documents(query_text, k=3, collection_name=COLLECTION_NAME):
    """查询相似文档"""
    # 获取集合
    collection = client.get_collection(name=collection_name)
    
    # 获取查询的嵌入向量
    query_embedding = embedding_model.embed_query(query_text)
    
    # 执行相似性搜索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # 将结果转换为Document对象
    documents = []
    if results['documents'] and len(results['documents']) > 0:
        for i, doc_text in enumerate(results['documents'][0]):
            if i < len(results['metadatas'][0]):
                metadata = results['metadatas'][0][i]
                documents.append(Document(page_content=doc_text, metadata=metadata))
    
    return documents

print("\n以上代码示例可以直接在Jupyter Notebook中使用")
print("添加新文档示例: add_documents_to_chroma(documents)")
print("查询相似文档示例: results = query_similar_documents('查询文本', k=3)") 