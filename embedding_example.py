#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
这是一个简单的示例脚本，演示如何正确创建embedding模型并与ChromaDB集成
"""

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    # 连接到ChromaDB服务
    print("正在连接ChromaDB服务...")
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    
    # 创建嵌入模型
    print("创建Embedding模型...")
    model_name = "thenlper/gte-large-zh"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # 创建测试文档
    print("准备测试文档...")
    documents = [
        Document(
            page_content="在向量搜索领域，我们拥有多种索引方法和向量处理技术，它们使我们能够在召回率、响应时间和内存使用之间做出权衡。",
            metadata={"source": "doc1", "category": "vector_search"}
        ),
        Document(
            page_content="虽然单独使用特定技术如倒排文件（IVF）、乘积量化（PQ）或分层导航小世界（HNSW）通常能够带来满意的结果",
            metadata={"source": "doc2", "category": "vector_search"}
        ),
        Document(
            page_content="GraphRAG 本质上就是 RAG，只不过与一般 RAG 相比，其检索路径上多了一个知识图谱",
            metadata={"source": "doc3", "category": "rag"}
        )
    ]
    
    # 使用LangChain的Chroma集成创建向量存储
    print("创建向量存储...")
    collection_name = "zhidu_db"
    embedding_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,  # 直接传递embedding对象，而不是调用get_embedding_fun()
        client=chroma_client,
        collection_name=collection_name
    )
    
    # 测试检索
    print("\n执行测试查询...")
    query = "索引技术有哪些？"
    results = embedding_db.similarity_search(query, k=2)
    
    print(f"\n查询: '{query}'")
    print("\n检索结果:")
    for i, doc in enumerate(results):
        print(f"\n结果 {i+1}:")
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
    
    print("\n完成测试!")

if __name__ == "__main__":
    main() 