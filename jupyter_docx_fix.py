"""
用于修复Jupyter Notebook中的docx导入问题
"""

import sys
import os
import importlib

def fix_docx_import():
    """修复Jupyter Notebook中docx导入问题"""
    # 检查是否已经有docx模块，如果有，检查它是否包含Document类
    if 'docx' in sys.modules:
        docx_module = sys.modules['docx']
        if hasattr(docx_module, 'Document'):
            print("docx模块已存在并包含Document类")
            return
        else:
            # 从sys.modules中移除错误的docx模块
            print("发现错误的docx模块，正在移除...")
            del sys.modules['docx']
    
    # 找到正确的python-docx包的路径
    correct_docx_path = None
    site_packages_dirs = [p for p in sys.path if 'site-packages' in p]
    
    for site_packages in site_packages_dirs:
        potential_path = os.path.join(site_packages, 'docx')
        if os.path.exists(potential_path) and os.path.isdir(potential_path):
            # 检查是否是正确的docx目录（应该有__init__.py和api.py文件）
            if os.path.exists(os.path.join(potential_path, '__init__.py')) and \
               os.path.exists(os.path.join(potential_path, 'api.py')):
                correct_docx_path = site_packages
                break
    
    if correct_docx_path:
        # 确保正确的路径在Python路径的最前面
        if correct_docx_path in sys.path:
            sys.path.remove(correct_docx_path)
        sys.path.insert(0, correct_docx_path)
        print(f"已将正确的docx路径添加到Python路径的前面: {correct_docx_path}")
        
        # 重新导入docx模块
        try:
            importlib.invalidate_caches()
            import docx
            print(f"成功导入docx模块: {docx.__file__}")
            if hasattr(docx, 'Document'):
                print("可以从docx直接导入Document类")
            else:
                print("警告: docx模块中没有Document类")
        except ImportError as e:
            print(f"导入docx模块失败: {e}")
    else:
        print("无法找到正确的docx模块路径")

# 在运行时自动执行修复
if __name__ == "__main__":
    fix_docx_import() 