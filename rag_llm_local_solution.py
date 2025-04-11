"""
RAG评估 - 使用本地文件系统版本

这个脚本使用本地文件系统模式而不是服务器模式来创建和使用Chroma数据库。
"""

# 确保安装tabulate
try:
    import tabulate
    print("tabulate已安装")
except ImportError:
    print("正在安装tabulate...")
    import subprocess
    subprocess.check_call(["pip", "install", "tabulate"])
    print("tabulate安装完成")

from model import RagEmbedding, RagLLM
from doc_parse import chunk, read_and_process_excel, logger
import pandas as pd
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import os
import shutil
from langchain_core.documents import Document

def main():
    print("## 设置参数")
    
    # 配置参数 - 使用本地文件系统模式
    PERSIST_DIRECTORY = "./chroma_db/zhidu_db"
    COLLECTION_NAME = "zhidu_db"
    
    # 如果目录已存在，删除它以确保重新创建
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    print(f"使用本地存储目录: {PERSIST_DIRECTORY}")
    
    pdf_files = ["./data/zhidu_employee.pdf", "./data/zhidu_travel.pdf"]
    excel_files = ["./data/zhidu_detail.xlsx"]
    
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
    
    print("\n## 处理PDF文件")
    
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
    
    print(f"处理了 {len(doc_data)} 个PDF文档片段")
    for i, chunk_text in enumerate(doc_data[:3]):
        print(f"\n示例 {i+1}:")
        print(len(chunk_text), "="*10, chunk_text)
    print("...等多个文档片段")
    
    print("\n## 处理Excel文件")
    
    for excel_file_name in excel_files:
        try:
            print(f"处理Excel文件: {excel_file_name}")
            data = read_and_process_excel(excel_file_name)
            df = pd.DataFrame(data[8:], columns=data[7])
            data_excel = df.drop(columns=df.columns[11:17])
            doc_data.append(data_excel.to_markdown(index=False).replace(' ', ""))
        except Exception as e:
            print(f"处理Excel文件失败: {str(e)}")
    
    print("\n## 创建文档对象")
    
    documents = []
    
    for chunk_text in doc_data:
        document = Document(
            page_content=chunk_text,
            metadata={"source": "test"})
        documents.append(document)
    
    print(f"总共创建了 {len(documents)} 个文档对象")
    
    print("\n## 初始化嵌入模型")
    
    embedding_cls = RagEmbedding(model_name="BAAI/bge-m3", device="cpu")
    
    print("\n## 创建Chroma数据库（使用本地文件系统模式）")
    
    embedding_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_cls,
        persist_directory=PERSIST_DIRECTORY,  # 使用本地目录存储
        collection_name=COLLECTION_NAME
    )
    
    # 注意：最新版本的Chroma不需要手动调用persist()方法
    # embedding_db.persist()  # 旧版本需要这一行
    print(f"成功创建Chroma数据库并保存至: {PERSIST_DIRECTORY}")
    
    print("\n## 测试相似性搜索")
    
    query = "迟到有什么规定？"
    print(f"查询: '{query}'")
    
    related_docs = embedding_db.similarity_search(query, k=2)
    
    print("最相关的文档:")
    for i, doc in enumerate(related_docs):
        print(f"\n{i+1}. {doc.page_content}")
        print(f"   来源: {doc.metadata['source']}")
    
    print("\n## 重新加载数据库（演示持久化后如何重新使用）")
    
    # 使用正确的方式重新加载已有的数据库
    loaded_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_cls,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # 获取集合中的文档数量
    collection_size = loaded_db._collection.count()
    print(f"成功重新加载数据库，包含 {collection_size} 个文档")
    
    query = "员工福利有哪些？"
    print(f"新查询: '{query}'")
    
    related_docs = loaded_db.similarity_search(query, k=2)
    
    print("最相关的文档:")
    for i, doc in enumerate(related_docs):
        print(f"\n{i+1}. {doc.page_content}")
        print(f"   来源: {doc.metadata['source']}")

if __name__ == "__main__":
    main() 