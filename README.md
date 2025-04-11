# RAG 检索增强生成系统

这个项目实现了一个基于本地部署的大语言模型(LLM)和检索增强生成(RAG)的文本处理系统。

## 功能特性

- 使用本地Ollama部署的Qwen模型
- 基于LangChain框架实现检索增强生成
- 支持Few-Shot提示模板
- 使用ChromaDB实现向量数据库

## 快速开始

### 环境准备

1. 确保安装了Ollama，可从[https://ollama.ai/](https://ollama.ai/)下载
2. 拉取Qwen:14b模型：`ollama pull qwen:14b`
3. 安装Python依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 检查环境

在使用前，建议运行检查工具验证Ollama服务是否正常：

```bash
python check_ollama.py
```

### 运行示例

可以运行测试脚本查看系统性能：

```bash
python test_notebook.py
```

或者直接运行Jupyter笔记本：

```bash
jupyter notebook rag_retrieval_chapter_9.ipynb
```

## 文件说明

- `model.py`: 实现了大语言模型和嵌入模型的接口
- `check_ollama.py`: Ollama状态检查工具
- `test_notebook.py`: 测试脚本，模拟笔记本的执行流程
- `rag_retrieval_chapter_9.ipynb`: 主要的RAG系统示例笔记本

## 常见问题及解决方案

### Ollama长时间无响应

**问题描述**: 在Jupyter Notebook中调用Ollama模型时，单元格执行长时间无响应。

**可能原因**:

1. **多个Ollama实例冲突**：同时运行多个Ollama会导致资源争用
2. **网络请求超时**：HTTP请求未设置超时时间
3. **系统资源不足**：大型模型(如Qwen:14b)需要较高的系统资源

**解决方案**:

1. 运行检查工具确认状态：`python check_ollama.py`
2. 关闭多余的Ollama实例：
   ```bash
   pkill -f "ollama run"
   ```
3. 在代码中添加超时控制和重试机制（已在`model.py`中实现）
4. 如仍然响应缓慢，考虑使用较小模型（如`qwen:7b`）或增加系统资源

### Jupyter Notebook卡住不动

**问题描述**: 执行含有模型调用的单元格时，Jupyter Notebook显示为"运行中"但长时间没有输出。

**解决方案**:

1. 按下两次Interrupt kernel按钮(I)停止执行
2. 重启内核(Restart)
3. 使用提供的测试脚本代替笔记本来诊断问题：`python test_notebook.py`
4. 确保使用最新版本的代码，包含超时控制和重试机制

## 性能优化建议

1. **减少模型大小**: 如果速度是关键，可以考虑使用`qwen:4b`或`qwen:7b`等较小模型
2. **优化提示模板**: 精简提示模板可以减少token数量，加快响应速度
3. **启用GPU加速**: 如果设备支持，可以尝试启用GPU来加速推理
4. **调整参数**:
   ```python
   # 在模型初始化时设置更短的超时和更多的重试
   llm = QwenLLM(timeout=10, max_retries=2)
   ```

## 相关资源

- [Ollama官方文档](https://github.com/ollama/ollama)
- [LangChain文档](https://python.langchain.com/docs)
- [Qwen模型仓库](https://github.com/QwenLM/Qwen) 