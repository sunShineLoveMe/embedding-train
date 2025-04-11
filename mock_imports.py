"""
这个模块提供了模拟的导入，以帮助测试doc_parse.py中的独立功能
"""
import sys
from unittest.mock import MagicMock

# 创建模拟模块
class MockModule(MagicMock):
    pass

# 创建模拟的api.db模块
api_db = MockModule()
api_db.ParserType = MockModule()
api_db.ParserType.MANUAL = MockModule()
api_db.ParserType.MANUAL.value = "manual"

# 创建模拟的rag.nlp模块
rag_nlp = MockModule()
rag_nlp.rag_tokenizer = MockModule()
rag_nlp.tokenize = MockModule()
rag_nlp.tokenize_table = MockModule()
rag_nlp.add_positions = MockModule()
rag_nlp.bullets_category = MockModule()
rag_nlp.title_frequency = MockModule()
rag_nlp.tokenize_chunks = MockModule()
rag_nlp.docx_question_level = MockModule()

# 创建模拟的rag.utils模块
rag_utils = MockModule()
rag_utils.num_tokens_from_string = lambda x: len(x) if isinstance(x, str) else 0

# 创建模拟的deepdoc.parser模块
deepdoc_parser = MockModule()
deepdoc_parser.PdfParser = MockModule
deepdoc_parser.PlainParser = MockModule
deepdoc_parser.ExcelParser = MockModule
deepdoc_parser.DocxParser = MockModule

# 注册模拟模块
sys.modules['api.db'] = api_db
sys.modules['rag.nlp'] = rag_nlp
sys.modules['rag.utils'] = rag_utils
sys.modules['deepdoc.parser'] = deepdoc_parser

# 模拟docx模块
docx = MockModule()
docx.Document = MockModule
sys.modules['docx'] = docx

# 只需要模拟PIL.Image，不需要实际实现
pil = MockModule()
pil.Image = MockModule()
sys.modules['PIL'] = pil
sys.modules['PIL.Image'] = pil.Image 