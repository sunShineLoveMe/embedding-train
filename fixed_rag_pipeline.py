"""
修复版RAG管道实现
解决了原始run_rag_pipline函数中的JSON解析错误问题
"""

from model import QwenLLM, RagLLM
from langchain.prompts import PromptTemplate
import time
import json

# 配置提示模板
prompt_template = """根据以下上下文，回答用户的问题。
    
上下文:
{context}

用户问题: {question}

请只回答上下文中包含的信息，如果上下文中没有相关信息，请说明"抱歉，我无法根据提供的上下文回答这个问题。"
"""

def fixed_run_rag_pipeline(query, context_query, k=3, context_query_type="query", 
                          stream=False, prompt_template=prompt_template,
                          temperature=0.1, llm=None, vector_db=None):
    """
    修复版RAG管道函数
    
    Args:
        query (str): 用户查询
        context_query (str or list): 用于检索的查询或文档
        k (int): 返回的相关文档数量
        context_query_type (str): 查询类型，可选值为"query"、"vector"或"doc"
        stream (bool): 是否使用流式输出
        prompt_template (str): 提示模板
        temperature (float): 模型温度参数
        llm (LangChain LLM): 语言模型实例，如果为None则创建新实例
        vector_db: 向量数据库实例，如果为None则使用传入的文档
        
    Returns:
        str: 模型回答
    """
    # 创建LLM实例（如果未提供）
    if llm is None:
        print("创建新的QwenLLM实例...")
        llm = QwenLLM(timeout=60, max_retries=3)
    
    # 处理上下文检索
    if context_query_type == "doc":
        # 直接使用提供的文档
        related_docs = context_query
        context = "\n".join([f"上下文{i+1}: {doc} \n" for i, doc in enumerate(related_docs)])
    else:
        # 没有向量数据库时直接使用context_query作为上下文
        if vector_db is None:
            print("警告: 没有提供向量数据库，直接使用context_query作为上下文")
            if isinstance(context_query, list):
                context = "\n".join([f"上下文{i+1}: {doc} \n" for i, doc in enumerate(context_query)])
            else:
                context = f"上下文: {context_query}"
        else:
            # 使用向量数据库检索
            if context_query_type == "vector":
                related_docs = vector_db.similarity_search_by_vector(context_query, k=k)
            else:  # "query"
                related_docs = vector_db.similarity_search(context_query, k=k)
            
            context = "\n".join([f"上下文{i+1}: {doc.page_content} \n" 
                              for i, doc in enumerate(related_docs)])
    
    # 打印调试信息
    print()
    print("#"*100)
    print(f"query: {query}")
    print(f"context: {context}")
    
    # 构建提示
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_template,
    )
    llm_prompt = prompt.format(question=query, context=context)
    
    # 使用语言模型生成回答
    try:
        start_time = time.time()
        
        if stream:
            print(f"response: ")
            response = llm(llm_prompt, stream=True, temperature=temperature)
            full_response = ""
            
            try:
                for chunk in response:
                    if isinstance(chunk, dict) and 'choices' in chunk:
                        text = chunk['choices'][0].get('text', '')
                    elif hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        text = chunk.choices[0].text
                    else:
                        text = str(chunk)
                    
                    print(text, end='', flush=True)
                    full_response += text
                
                print()  # 添加换行
                elapsed = time.time() - start_time
                print(f"完成，耗时: {elapsed:.2f}秒")
                return full_response
            except Exception as e:
                print(f"\n流式输出处理错误: {str(e)}")
                # 失败时回退到非流式模式
                return llm(llm_prompt, stream=False, temperature=temperature)
        else:
            # 非流式模式
            response = llm(llm_prompt, stream=False, temperature=temperature)
            elapsed = time.time() - start_time
            print(f"response: {response}")
            print(f"完成，耗时: {elapsed:.2f}秒")
            return response
            
    except Exception as e:
        print(f"错误: {str(e)}")
        # 在发生错误时提供错误说明
        return f"抱歉，处理您的请求时发生错误: {str(e)}"

# 使用示例
if __name__ == "__main__":
    # 示例查询和上下文
    query = "那个，我们公司有什么规定来着？"
    context = [
        "1、学校的工作时间由学校决定并公布。学校内除特殊岗位特别规定外，全体教职员工均应严格执行学校的作息时间表，不迟到、不早退、不中途离校",
        "1、学校的工作时间由学校决定并公布。学校内除特殊岗位特别规定外，全体教职员工均应严格执行学校的作息时间表，不迟到、不早退、不中途离校",
        "企业形象是学校非常重要的财富，维护好企业形象是每个员工必须遵守的规则。员工必须严格遵守学校的企业文化，经营理念和管理制度，充分维护和支持学校的企业形象建设。任何人不得出现有损学校团队建设、诋毁学校企业管理和企业文化的语言行为"
    ]
    
    # 使用修复后的函数
    result = fixed_run_rag_pipeline(
        query=query,
        context_query=context,
        context_query_type="doc",
        stream=False
    )
    
    print("\n最终结果:")
    print(result) 