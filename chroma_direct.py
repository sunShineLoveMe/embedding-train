"""
使用ChromaDB底层API直接创建和使用向量数据库
此方法避开了langchain_chroma层可能存在的问题
"""

import chromadb
from model import RagEmbedding
import numpy as np
from typing import List, Dict, Any
from langchain_core.documents import Document

# Chroma服务器连接信息
CHROMA_SERVER_HOST = "localhost"
CHROMA_SERVER_PORT = "8000"  # 默认端口，如果你修改过，请使用实际端口
COLLECTION_NAME = "direct_db"

def create_sample_documents():
    """创建一些示例文档"""
    return [
        Document(page_content="中国是一个有着悠久历史的国家", metadata={"source": "中国简介"}),
        Document(page_content="北京是中国的首都，是政治和文化中心", metadata={"source": "北京简介"}),
        Document(page_content="上海是中国最大的经济中心城市", metadata={"source": "上海简介"}),
        Document(page_content="广州是中国南方重要的商业城市", metadata={"source": "广州简介"}),
        Document(page_content="深圳是中国重要的科技创新中心", metadata={"source": "深圳简介"}),
    ]

class ChromaDirectDB:
    """直接使用ChromaDB API的包装类"""
    
    def __init__(self, host: str, port: str, collection_name: str, embedding_model: RagEmbedding):
        """初始化ChromaDB客户端和集合"""
        self.client = chromadb.HttpClient(host=host, port=port)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # 检查并删除已存在的同名集合
        try:
            existing_collections = self.client.list_collections()
            for collection in existing_collections:
                if collection.name == collection_name:
                    self.client.delete_collection(collection_name)
                    print(f"已删除现有的集合: {collection_name}")
                    break
        except Exception as e:
            print(f"检查集合时出错: {str(e)}")
            
        # 创建新集合
        self.collection = self.client.create_collection(name=collection_name)
        print(f"成功创建集合: {collection_name}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到集合"""
        texts = [doc.page_content for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]
        
        # 使用嵌入模型将文本转换为向量
        embeddings = self.embedding_model.embed_documents(texts)
        
        # 将数据添加到集合
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print(f"成功添加 {len(documents)} 个文档到集合")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """执行相似性搜索并返回Document对象"""
        # 使用嵌入模型将查询转换为向量
        query_embedding = self.embedding_model.embed_query(query)
        
        # 使用向量进行相似性搜索
        results = self.collection.query(
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

def main():
    """主函数：测试直接使用ChromaDB API"""
    print("步骤1: 准备环境和数据")
    documents = create_sample_documents()
    print(f"创建了 {len(documents)} 个示例文档")
    
    print("\n步骤2: 初始化嵌入模型")
    embedding_model = RagEmbedding(model_name="BAAI/bge-m3", device="cpu")
    
    print("\n步骤3: 创建ChromaDB集合")
    try:
        db = ChromaDirectDB(
            host=CHROMA_SERVER_HOST,
            port=CHROMA_SERVER_PORT,
            collection_name=COLLECTION_NAME,
            embedding_model=embedding_model
        )
    except Exception as e:
        print(f"创建ChromaDB集合失败: {str(e)}")
        print("请确保Chroma服务器已启动，并可通过 http://localhost:8000 访问")
        return
    
    print("\n步骤4: 添加文档到集合")
    try:
        db.add_documents(documents)
    except Exception as e:
        print(f"添加文档失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n步骤5: 测试相似性搜索")
    try:
        query = "中国的城市"
        results = db.similarity_search(query, k=3)
        print(f"查询: '{query}'")
        print("相似度最高的3个文档:")
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.page_content} (来源: {doc.metadata['source']})")
    except Exception as e:
        print(f"相似性搜索失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n步骤6: 重新连接到已有集合")
    try:
        # 重新创建一个客户端并获取已有集合
        client = chromadb.HttpClient(host=CHROMA_SERVER_HOST, port=CHROMA_SERVER_PORT)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"成功连接到已有集合: {COLLECTION_NAME}")
        
        # 测试集合中的内容
        item_count = collection.count()
        print(f"集合中包含 {item_count} 个文档")
    except Exception as e:
        print(f"重新连接到集合失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 