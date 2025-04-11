"""
使用Chroma本地文件系统模式解决RAG系统向量数据库问题
这个脚本演示如何使用本地文件系统模式而非服务器模式来创建和使用Chroma数据库
"""

from model import RagEmbedding
from doc_parse import chunk, read_and_process_excel, logger
import pandas as pd
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import shutil

# 配置参数
PERSIST_DIRECTORY = "./chroma_db/zhidu_db"
COLLECTION_NAME = "zhidu_db"
pdf_files = ["./data/zhidu_employee.pdf", "./data/zhidu_travel.pdf"]
excel_files = ["./data/zhidu_detail.xlsx"]

def main():
    """主函数：使用本地文件系统模式创建和使用Chroma数据库"""
    print("步骤1: 准备环境")
    
    # 如果目录已存在，删除它以确保重新创建
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # 确保安装tabulate包，用于处理Excel数据
    try:
        import tabulate
    except ImportError:
        print("正在安装缺少的tabulate依赖...")
        import subprocess
        subprocess.check_call(["pip", "install", "tabulate"])
        print("tabulate安装完成")
    
    print("\n步骤2: 处理文档")
    
    # 配置文本分割器
    r_spliter = RecursiveCharacterTextSplitter(
        chunk_size=128,
        chunk_overlap=30,
        separators=["\n\n", 
                   "\n", 
                   ".", 
                   "\uff0e", 
                   "\u3002",
                   ",",
                   "\uff0c",
                   "\u3001'"
                   ])
    
    # 处理PDF文件
    doc_data = []
    for pdf_file_name in pdf_files:
        try:
            print(f"处理PDF文件: {pdf_file_name}")
            res = chunk(pdf_file_name, callback=logger)
            for data in res:
                content = data["content_with_weight"]
                if '<table>' not in content and len(content) > 200:
                    doc_data = doc_data + r_spliter.split_text(content)
                else:
                    doc_data.append(content)
        except Exception as e:
            print(f"处理PDF文件失败: {str(e)}")
    
    # 处理Excel文件
    for excel_file_name in excel_files:
        try:
            print(f"处理Excel文件: {excel_file_name}")
            data = read_and_process_excel(excel_file_name)
            df = pd.DataFrame(data[8:], columns=data[7])
            data_excel = df.drop(columns=df.columns[11:17])
            doc_data.append(data_excel.to_markdown(index=False).replace(' ', ""))
        except Exception as e:
            print(f"处理Excel文件失败: {str(e)}")
    
    # 创建Document对象
    documents = []
    for chunk_text in doc_data:
        document = Document(
            page_content=chunk_text,
            metadata={"source": "test"}
        )
        documents.append(document)
    
    print(f"总共处理了 {len(documents)} 个文档片段")
    
    print("\n步骤3: 初始化嵌入模型")
    try:
        embedding_cls = RagEmbedding(model_name="BAAI/bge-m3", device="cpu")
    except Exception as e:
        print(f"初始化嵌入模型失败: {str(e)}")
        return
    
    print("\n步骤4: 创建Chroma数据库（本地文件系统模式）")
    try:
        embedding_db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_cls,
            persist_directory=PERSIST_DIRECTORY,  # 使用本地目录存储
            collection_name=COLLECTION_NAME
        )
        
        # 注意：不再调用persist()方法，因为新版本的Chroma会自动持久化
        print(f"成功创建Chroma数据库并保存至: {PERSIST_DIRECTORY}")
    except Exception as e:
        print(f"创建Chroma数据库失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n步骤5: 测试相似性搜索")
    try:
        query = "迟到有什么规定？"
        print(f"查询: '{query}'")
        
        related_docs = embedding_db.similarity_search(query, k=2)
        
        print("最相关的文档:")
        for i, doc in enumerate(related_docs):
            print(f"{i+1}. {doc.page_content}")
            print(f"   来源: {doc.metadata['source']}")
            print()
    except Exception as e:
        print(f"相似性搜索失败: {str(e)}")
    
    print("\n步骤6: 重新加载数据库")
    try:
        # 使用正确的方式重新加载已有的数据库
        loaded_db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_cls,
            persist_directory=PERSIST_DIRECTORY
        )
        
        # 获取集合中的文档数量
        collection_size = loaded_db._collection.count()
        print(f"成功重新加载数据库，包含 {collection_size} 个文档")
        
        # 再次测试查询
        query = "员工福利有哪些？"
        print(f"新查询: '{query}'")
        
        related_docs = loaded_db.similarity_search(query, k=2)
        
        print("最相关的文档:")
        for i, doc in enumerate(related_docs):
            print(f"{i+1}. {doc.page_content}")
            print(f"   来源: {doc.metadata['source']}")
            print()
    except Exception as e:
        print(f"重新加载数据库失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 