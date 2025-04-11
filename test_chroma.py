import chromadb
from model import RagEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import shutil

# 测试目录
DB_DIR = "./test_chroma_db"

def setup():
    """设置测试环境"""
    # 如果测试目录已存在，则删除
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)
    
    # 创建一些测试文档
    documents = [
        Document(page_content="这是第一个测试文档", metadata={"source": "test1"}),
        Document(page_content="这是第二个测试文档", metadata={"source": "test2"}),
        Document(page_content="这是第三个测试文档", metadata={"source": "test3"}),
    ]
    return documents

def test_chroma_creation():
    """测试Chroma数据库创建"""
    print("步骤1: 设置测试环境")
    documents = setup()
    
    print("\n步骤2: 初始化嵌入模型")
    embedding_cls = RagEmbedding(model_name="BAAI/bge-m3")
    
    print("\n步骤3: 创建Chroma数据库（不使用客户端）")
    try:
        # 使用新的创建方式，不需要显式创建客户端
        embedding_db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_cls,
            persist_directory=DB_DIR,
            collection_name="test_collection"
        )
        print("成功创建Chroma数据库!")
        
        # 测试简单的相似性搜索
        results = embedding_db.similarity_search("测试", k=1)
        print(f"\n相似性搜索结果: {results}")
    except Exception as e:
        print(f"创建Chroma数据库失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n步骤4: 清理测试环境")
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

if __name__ == "__main__":
    test_chroma_creation() 