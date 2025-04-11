import sys
import os

print("Python路径:")
for path in sys.path:
    print(f"- {path}")

# 检查是否有docx.py或docx文件夹
for path in sys.path:
    docx_py = os.path.join(path, "docx.py")
    docx_dir = os.path.join(path, "docx")
    
    if os.path.exists(docx_py):
        print(f"找到docx.py文件: {docx_py}")
    
    if os.path.exists(docx_dir) and os.path.isdir(docx_dir):
        print(f"找到docx目录: {docx_dir}")

# 尝试导入docx
try:
    import docx
    print(f"\ndocx模块路径: {docx.__file__}")
    print(f"docx模块目录: {os.path.dirname(docx.__file__)}")
    print(f"docx模块内容: {dir(docx)}")
    
    if hasattr(docx, 'Document'):
        print("可以从docx直接导入Document")
    else:
        print("docx模块中没有Document")
        
        # 检查api子模块
        if hasattr(docx, 'api'):
            try:
                from docx.api import Document
                print("可以从docx.api导入Document")
            except ImportError as e:
                print(f"从docx.api导入Document失败: {e}")
        else:
            print("docx没有api子模块")
            
except ImportError as e:
    print(f"导入docx失败: {e}") 