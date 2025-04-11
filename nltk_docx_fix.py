"""
全面修复NLTK数据问题和docx导入问题的脚本

在Jupyter Notebook中，如果遇到NLTK数据问题或docx导入问题，请在notebook开头导入并运行此脚本中的fix_all()函数。
"""

import os
import sys
import importlib
import subprocess
import platform

def fix_nltk_data():
    """修复NLTK数据问题"""
    print("开始修复NLTK数据问题...")
    
    try:
        import nltk
        
        # 完全清除NLTK现有的数据路径
        nltk.data.path = []
        
        # 设置新的数据路径到当前工作目录下的nltk_data文件夹
        current_dir = os.getcwd()
        nltk_data_dir = os.path.join(current_dir, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        print(f"设置NLTK数据路径为: {nltk_data_dir}")
        
        # 下载常用NLTK数据包
        packages = ['punkt', 'wordnet', 'omw-1.4', 'stopwords']
        for package in packages:
            print(f"正在下载 {package}...")
            nltk.download(package, download_dir=nltk_data_dir)
        
        # 检查word_tokenize是否正常工作
        try:
            from nltk.tokenize import word_tokenize
            test_result = word_tokenize("测试tokenize是否正常工作")
            print(f"word_tokenize测试成功: {test_result}")
        except Exception as e:
            print(f"word_tokenize测试失败: {e}")
            
            # 如果失败，尝试修复rag_tokenizer.py文件
            try:
                fix_rag_tokenizer()
            except Exception as e:
                print(f"无法修复rag_tokenizer.py: {e}")
                print("请手动修改rag_tokenizer.py文件的tokenize方法")
        
        print("NLTK数据问题修复完成")
        return True
    except Exception as e:
        print(f"修复NLTK数据问题时出错: {e}")
        return False

def fix_rag_tokenizer():
    """修改rag_tokenizer.py文件，避免依赖word_tokenize"""
    try:
        import re
        
        # 查找rag_tokenizer.py文件
        current_dir = os.getcwd()
        rag_tokenizer_path = os.path.join(current_dir, "rag/nlp/rag_tokenizer.py")
        
        if not os.path.exists(rag_tokenizer_path):
            print(f"找不到rag_tokenizer.py文件: {rag_tokenizer_path}")
            return False
        
        # 读取文件内容
        with open(rag_tokenizer_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换tokenize方法
        pattern = r'def tokenize\(self, line\):.*?def fine_grained_tokenize\(self, tks\):'
        replacement = '''def tokenize(self, line):
        """
        分词
        """
        line = self._strQ2B(line).lower()
        line = self._tradi2simp(line)
        
        if not line.strip():
            return ""
        
        # 检查是否是英文文本
        zh_num = len([1 for c in line if is_chinese(c)])
        if zh_num == 0:
            # 替换原来的word_tokenize调用，使用更简单的方法
            try:
                # 尝试使用原始的word_tokenize
                from nltk.tokenize import word_tokenize
                return " ".join([self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in word_tokenize(line)])
            except Exception as e:
                # 如果word_tokenize失败，使用简单的空格分割和标点符号处理
                import re
                # 在标点符号前后添加空格
                line = re.sub(r'([.,!?;:])', r' \\1 ', line)
                # 分割成单词并进行词形还原和词干提取
                tokens = [t.strip() for t in line.split() if t.strip()]
                return " ".join([self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in tokens])
        
        arr = re.split(self.SPLIT_CHAR, line)
        res = []
        for L in arr:
            if len(L) < 2 or re.match(
                    r"[a-z\\.-]+$", L) or re.match(r"[0-9\\.-]+$", L):
                res.append(L)
                continue
            # print(L)

            # use maxforward for the first time
            tks, s = self.maxForward_(L)
            tks1, s1 = self.maxBackward_(L)
            if self.DEBUG:
                print("[FW]", tks, s)
                print("[BW]", tks1, s1)

            diff = [0 for _ in range(max(len(tks1), len(tks)))]
            for i in range(min(len(tks1), len(tks))):
                if tks[i] != tks1[i]:
                    diff[i] = 1

            if s1 > s:
                tks = tks1

            i = 0
            while i < len(tks):
                s = i
                while s < len(tks) and diff[s] == 0:
                    s += 1
                if s == len(tks):
                    res.append(" ".join(tks[i:]))
                    break
                if s > i:
                    res.append(" ".join(tks[i:s]))

                e = s
                while e < len(tks) and e - s < 5 and diff[e] == 1:
                    e += 1

                tkslist = []
                self.dfs_("".join(tks[s:e + 1]), 0, [], tkslist)
                res.append(" ".join(self.sortTks_(tkslist)[0][0]))

                i = e + 1

        res = " ".join(self.english_normalize_(res))
        if self.DEBUG:
            print("[TKS]", self.merge_(res))
        return self.merge_(res)

    def fine_grained_tokenize(self, tks):'''
        
        # 使用正则表达式的DOTALL模式，确保可以匹配跨越多行的内容
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # 写回文件
        with open(rag_tokenizer_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"成功修改rag_tokenizer.py文件")
        return True
    except Exception as e:
        print(f"修改rag_tokenizer.py文件时出错: {e}")
        return False

def fix_docx_import():
    """修复docx导入问题"""
    print("开始修复docx导入问题...")
    
    try:
        # 检查是否已经有docx模块，如果有，检查它是否包含Document类
        if 'docx' in sys.modules:
            docx_module = sys.modules['docx']
            if hasattr(docx_module, 'Document'):
                print("docx模块已存在并包含Document类")
                return True
            else:
                # 从sys.modules中移除错误的docx模块
                print("发现错误的docx模块，正在移除...")
                del sys.modules['docx']
        
        # 确保python-docx正确安装
        try:
            import docx
            print(f"成功导入docx模块: {docx.__file__}")
            if hasattr(docx, 'Document'):
                print("可以从docx直接导入Document类")
                return True
        except ImportError:
            print("无法导入docx模块，尝试重新安装python-docx...")
            
            # 尝试安装python-docx
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "python-docx"])
            
            # 重新导入
            importlib.invalidate_caches()
            try:
                import docx
                print(f"成功导入docx模块: {docx.__file__}")
                if hasattr(docx, 'Document'):
                    print("可以从docx直接导入Document类")
                    return True
            except ImportError as e:
                print(f"重新安装后仍无法导入docx模块: {e}")
        
        print("docx导入问题修复完成")
        return True
    except Exception as e:
        print(f"修复docx导入问题时出错: {e}")
        return False

def fix_libomp():
    """修复libomp库问题（仅在macOS上）"""
    if platform.system() != 'Darwin':
        print("非macOS系统，不需要修复libomp库")
        return True
    
    print("开始修复libomp库问题...")
    
    try:
        # 检查是否安装了libomp
        result = subprocess.run(["brew", "list", "libomp"], capture_output=True, text=True)
        if result.returncode != 0:
            print("未安装libomp库，尝试安装...")
            subprocess.check_call(["brew", "install", "libomp"])
        else:
            print("已安装libomp库")
        
        # 检查libomp.dylib文件是否存在
        libomp_path = "/opt/homebrew/lib/libomp.dylib"
        if not os.path.exists(libomp_path):
            print("libomp.dylib文件不存在，尝试创建符号链接...")
            
            # 寻找已安装的libomp.dylib
            find_result = subprocess.run([
                "find", 
                "/opt/homebrew/Cellar/libomp", 
                "-name", "libomp.dylib"
            ], capture_output=True, text=True)
            
            if find_result.returncode != 0 or not find_result.stdout.strip():
                print("找不到libomp.dylib文件")
                return False
            
            source_path = find_result.stdout.strip().split("\n")[0]
            print(f"找到libomp.dylib文件: {source_path}")
            
            # 创建符号链接
            os.makedirs(os.path.dirname(libomp_path), exist_ok=True)
            os.symlink(source_path, libomp_path)
            print(f"成功创建符号链接: {source_path} -> {libomp_path}")
        else:
            print(f"libomp.dylib文件已存在: {libomp_path}")
        
        print("libomp库问题修复完成")
        return True
    except Exception as e:
        print(f"修复libomp库问题时出错: {e}")
        return False

def fix_imports():
    """修复导入问题，确保RagEmbedding类可以被导入"""
    print("开始修复导入问题...")
    
    try:
        # 使用sys.path.append而不是修改PYTHONPATH
        current_dir = os.getcwd()
        
        # 添加当前目录到Python路径
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            print(f"将当前目录 {current_dir} 添加到Python路径")
        
        # 检查是否可以导入RagEmbedding
        try:
            # 尝试直接导入
            exec("from model import RagEmbedding")
            print("成功导入RagEmbedding类")
            return True
        except ImportError:
            print("无法直接导入RagEmbedding类，尝试其他修复方法...")
        
        # 检查model.py文件是否存在
        model_path = os.path.join(current_dir, "model.py")
        if not os.path.exists(model_path):
            print(f"找不到model.py文件: {model_path}")
            print("提示: RagEmbedding类应该在model.py文件中")
            print("请确保该文件在当前目录中")
            return False
        
        # 重新加载模块
        if 'model' in sys.modules:
            print("重新加载model模块...")
            importlib.reload(sys.modules['model'])
        
        print("导入问题修复完成")
        return True
    except Exception as e:
        print(f"修复导入问题时出错: {e}")
        return False

def fix_all():
    """修复所有问题"""
    print("开始全面修复...")
    
    # 修复导入问题
    if not fix_imports():
        print("警告：导入问题修复失败")
    
    # 修复docx导入问题
    if not fix_docx_import():
        print("警告：docx导入问题修复失败")
    
    # 修复NLTK数据问题
    if not fix_nltk_data():
        print("警告：NLTK数据问题修复失败")
    
    # 修复libomp库问题（仅在macOS上）
    if platform.system() == 'Darwin':
        if not fix_libomp():
            print("警告：libomp库问题修复失败")
    
    print("全面修复完成！现在可以尝试导入所需模块。")
    print("\n使用示例：")
    print("from model import RagEmbedding")
    print("from doc_parse import chunk, read_and_process_excel, logger")
    
    return True

if __name__ == "__main__":
    fix_all() 