from model import RagEmbedding
import langchain_chroma as lc_chroma
from langchain_core.documents import Document

def test_embedding_model():
    """测试嵌入模型的基础功能"""
    print("测试1: 初始化嵌入模型")
    embedding_cls = RagEmbedding(model_name="BAAI/bge-m3")
    
    print("\n测试2: 嵌入单个文本")
    test_text = "这是一个测试文本"
    try:
        embedding = embedding_cls.embed_query(test_text)
        print(f"成功生成嵌入向量，维度: {len(embedding)}")
    except Exception as e:
        print(f"嵌入失败: {str(e)}")
    
    print("\n测试3: 嵌入多个文本")
    test_texts = ["这是第一个测试文本", "这是第二个测试文本"]
    try:
        embeddings = embedding_cls.embed_documents(test_texts)
        print(f"成功生成嵌入向量，数量: {len(embeddings)}, 每个维度: {len(embeddings[0])}")
    except Exception as e:
        print(f"嵌入失败: {str(e)}")
    
    print("\n测试4: 检查Chroma兼容性")
    try:
        # 打印出RagEmbedding类的属性和方法
        print("RagEmbedding类的属性和方法:")
        for attr in dir(embedding_cls):
            if not attr.startswith('__'):
                print(f"  - {attr}")
        
        # 检查是否是Embeddings类的实例
        from langchain.embeddings.base import Embeddings
        print(f"\n是Embeddings实例: {isinstance(embedding_cls, Embeddings)}")
        
        # 创建一个测试文档
        test_doc = Document(page_content="这是测试文档内容", metadata={"source": "test"})
        
        # 从Chroma中导入必要的类
        from langchain_chroma import Chroma
        
        # 打印Chroma.from_documents的参数
        print("\nChroma.from_documents的参数:")
        import inspect
        sig = inspect.signature(Chroma.from_documents)
        for param_name, param in sig.parameters.items():
            print(f"  - {param_name}: {param.annotation}")
    except Exception as e:
        print(f"兼容性检查失败: {str(e)}")

if __name__ == "__main__":
    test_embedding_model() 