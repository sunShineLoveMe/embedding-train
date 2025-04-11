#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG系统使用示例

本脚本演示了如何使用修复后的RAG系统进行文档处理、向量嵌入和检索问答。
"""

import os
import warnings
warnings.filterwarnings('ignore')

# 导入修复后的模块
from model import RagLLM, RagEmbedding
from doc_parse import chunk, read_and_process_excel, logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def main():
    print("=" * 50)
    print("RAG系统使用示例")
    print("=" * 50)
    
    # 确保数据目录存在
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./chroma_db_example", exist_ok=True)
    
    # 检查示例PDF文件是否存在
    pdf_files = ["./data/zhidu_employee.pdf", "./data/zhidu_travel.pdf"]
    available_files = []
    
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            available_files.append(pdf_file)
            print(f"找到文件: {pdf_file}")
        else:
            print(f"文件不存在: {pdf_file}，请确保您有权限访问此文件或将正确的PDF文件放在data目录中")
    
    if not available_files:
        print("未找到可用的PDF文件，将使用示例文本进行演示")
        # 使用一些示例文本
        sample_texts = [
            "员工请假规定：员工请假需提前3天向部门主管提交书面申请。病假需提供医院证明。年假每年20天，可分次使用。",
            "出差报销制度：出差期间的交通、住宿和餐饮费用可报销。需保留发票和行程记录。机票需经部门主管审批。",
            "工作时间规定：正常工作时间为周一至周五9:00-18:00，午休12:00-13:00。加班需提前申请并获得批准。"
        ]
        
        # 初始化文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=128,
            chunk_overlap=30,
            separators=["\n\n", "\n", "。", "，"]
        )
        
        # 直接使用示例文本
        doc_data = sample_texts
        
    else:
        print("\n开始处理PDF文件...")
        # 处理PDF文件
        doc_data = []
        
        # 初始化文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=128,
            chunk_overlap=30,
            separators=["\n\n", "\n", ".", "。", ",", "，", "；"]
        )
        
        # 为了简化示例，只处理第一个可用的PDF文件
        pdf_file = available_files[0]
        
        try:
            print(f"处理文件: {pdf_file}")
            res = chunk(pdf_file, callback=logger)
            
            for data in res:
                content = data.get("content_with_weight", "")
                if content and '<table>' not in content and len(content) > 100:
                    # 分割较长文本
                    chunks = text_splitter.split_text(content)
                    doc_data.extend(chunks)
                else:
                    doc_data.append(content)
            
            print(f"成功提取 {len(doc_data)} 个文本片段")
            
        except Exception as e:
            print(f"处理PDF时出错: {e}")
            # 使用一些示例文本
            doc_data = [
                "员工请假规定：员工请假需提前3天向部门主管提交书面申请。病假需提供医院证明。年假每年20天，可分次使用。",
                "出差报销制度：出差期间的交通、住宿和餐饮费用可报销。需保留发票和行程记录。机票需经部门主管审批。",
                "工作时间规定：正常工作时间为周一至周五9:00-18:00，午休12:00-13:00。加班需提前申请并获得批准。"
            ]
    
    # 显示部分文本内容
    print("\n文本示例:")
    for i, text in enumerate(doc_data[:3]):  # 只显示前3个
        print(f"[{i+1}] {text[:100]}..." if len(text) > 100 else f"[{i+1}] {text}")
    
    print("\n创建向量数据库...")
    try:
        # 初始化嵌入模型
        embedding_model = RagEmbedding().get_embedding_fun()
        
        # 创建向量数据库 - 限制处理量
        sample_size = min(10, len(doc_data))  # 最多处理10个文档，避免API请求过多
        print(f"向量化 {sample_size} 个文档...")
        
        vectordb = Chroma.from_texts(
            texts=doc_data[:sample_size],
            embedding=embedding_model,
            persist_directory="./chroma_db_example"
        )
        
        print("向量数据库创建成功")
        
        # 执行示例查询
        print("\n执行示例查询...")
        queries = ["员工请假规定", "出差报销政策", "工作时间是什么"]
        
        for query in queries:
            print(f"\n查询: '{query}'")
            # 执行相似性搜索
            docs = vectordb.similarity_search(query, k=2)
            
            print("检索结果:")
            for i, doc in enumerate(docs):
                print(f"结果 {i+1}: {doc.page_content[:150]}..." if len(doc.page_content) > 150 else f"结果 {i+1}: {doc.page_content}")
            
            # 使用LLM生成回答
            print("\n生成回答...")
            try:
                # 初始化大语言模型
                llm = RagLLM()
                
                # 构建提示词
                context = "\n\n".join([doc.page_content for doc in docs])
                prompt = f"""请根据以下信息回答问题:
                        信息:
                        {context}

                        问题: {query}
                        请提供简洁、准确的回答，只基于上述提供的信息。
                        回答:"""                
                # 生成回答
                response = llm(prompt, temperature=0.3)
                
                print(f"AI回答: {response}")
                
            except Exception as e:
                print(f"生成回答时出错: {e}")

            print("-" * 50)

    except Exception as e:
        print(f"创建向量数据库或执行查询时出错: {e}")
    
if __name__ == "__main__":
    main() 