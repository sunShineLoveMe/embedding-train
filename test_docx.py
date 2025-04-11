try:
    from docx import Document
    print("成功导入Document类！", Document)
except ImportError as e:
    print(f"导入错误: {e}")
    
    # 尝试通过完整路径导入
    try:
        import sys
        print("Python路径:", sys.path)
        
        # 尝试不同的导入方式
        import docx
        print("docx模块:", docx)
        print("docx模块路径:", docx.__file__)
        
        # 尝试直接从包导入
        from docx.api import Document as Document2
        print("成功通过api导入Document类！", Document2)
    except ImportError as e2:
        print(f"第二次导入错误: {e2}") 