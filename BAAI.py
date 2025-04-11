from transformers import AutoModel, AutoTokenizer

model_name = "BAAI/bge-m3"
# 自动下载模型和分词器
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)