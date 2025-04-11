import nltk
import os

# 检查并下载所需的NLTK数据
nltk_data_path = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)

# 设置NLTK数据路径
nltk.data.path.append(nltk_data_path)

# 检查并下载必要的NLTK数据集
required_packages = ['punkt', 'wordnet', 'omw-1.4', 'stopwords']
for package in required_packages:
    try:
        nltk.data.find(f'tokenizers/{package}')
        print(f"{package} 数据包已存在")
    except LookupError:
        print(f"正在下载 {package} 数据包...")
        nltk.download(package, download_dir=nltk_data_path)
        print(f"{package} 数据包下载完成")

# 解决docx模块问题
import sys
if 'docx' in sys.modules:
    print(f"移除已加载的docx模块")
    del sys.modules['docx']