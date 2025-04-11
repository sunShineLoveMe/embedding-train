"""
解决方案：创建和使用Chroma向量数据库

这个文件展示了如何正确地使用修改后的RagEmbedding类创建Chroma向量数据库，
并进行简单的相似性搜索。
连接到本地运行的Chroma服务器实例，而不是使用文件系统。
"""

from model import RagEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import shutil
import chromadb

# Chroma服务器连接信息
CHROMA_SERVER_HOST = "localhost"
CHROMA_SERVER_PORT = "8000"  # 默认端口，如果你修改过，请使用实际端口
COLLECTION_NAME = "zhidu_db"

def create_sample_documents():
    """创建一些示例文档"""
    return [
        Document(page_content="中国是一个有着悠久历史的国家", metadata={"source": "中国简介"}),
        Document(page_content="北京是中国的首都，是政治和文化中心", metadata={"source": "北京简介"}),
        Document(page_content="上海是中国最大的经济中心城市", metadata={"source": "上海简介"}),
        Document(page_content="广州是中国南方重要的商业城市", metadata={"source": "广州简介"}),
        Document(page_content="深圳是中国重要的科技创新中心", metadata={"source": "深圳简介"}),
    ]

def main():
    """主函数：创建并使用Chroma数据库"""
    print("步骤1: 准备环境和数据")
    
    # 创建示例文档
    documents = create_sample_documents()
    print(f"创建了 {len(documents)} 个示例文档")
    
    print("\n步骤2: 初始化嵌入模型")
    embedding_cls = RagEmbedding(model_name="BAAI/bge-m3", device="cpu")
    
    print("\n步骤3: 连接到Chroma服务器")
    try:
        # 创建HTTP客户端连接到本地Chroma服务器
        chroma_client = chromadb.HttpClient(
            host=CHROMA_SERVER_HOST,
            port=CHROMA_SERVER_PORT
        )
        print(f"成功连接到Chroma服务器: http://{CHROMA_SERVER_HOST}:{CHROMA_SERVER_PORT}")
        
        # 删除已存在的同名集合（如果存在）
        try:
            existing_collections = chroma_client.list_collections()
            for collection in existing_collections:
                if collection.name == COLLECTION_NAME:
                    chroma_client.delete_collection(COLLECTION_NAME)
                    print(f"已删除现有的集合: {COLLECTION_NAME}")
                    break
        except Exception as e:
            print(f"检查集合时出错: {str(e)}")
    except Exception as e:
        print(f"连接到Chroma服务器失败: {str(e)}")
        print("请确保Chroma服务器已启动，并可通过 http://localhost:8000 访问")
        return
    
    print("\n步骤4: 创建Chroma数据库")
    try:
        # 使用HTTP客户端创建Chroma数据库
        embedding_db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_cls,
            client=chroma_client,
            collection_name=COLLECTION_NAME
        )
        print("成功创建Chroma数据库!")
        
        # 测试相似性搜索
        query = "中国的城市"
        results = embedding_db.similarity_search(query, k=3)
        print(f"\n查询: '{query}'")
        print("相似度最高的3个文档:")
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.page_content} (来源: {doc.metadata['source']})")
    except Exception as e:
        print(f"创建Chroma数据库失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n步骤5: 再次连接到已有集合")
    try:
        # 重新连接到已有集合
        loaded_db = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_cls
        )
        print(f"成功连接到已有集合: {COLLECTION_NAME}")
        
        # 测试重新连接的数据库
        results = loaded_db.similarity_search("北京", k=1)
        print(f"从已有集合中检索结果: {results[0].page_content}")
    except Exception as e:
        print(f"连接到已有集合失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 