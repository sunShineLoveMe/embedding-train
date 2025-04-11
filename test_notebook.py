from model import QwenLLM
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import time

# 记录总运行时间
total_start = time.time()

print("===== 初始化模型 =====")
start = time.time()
langchain_llm = QwenLLM()
print(f"初始化耗时: {time.time()-start:.2f}秒")

print("\n===== 测试1: 简单提示模板 =====")
start = time.time()
template = """
    {our_text}
    你能为上述内容创建一个包含 {wordsCount} 个词的推文吗？
"""
print(f"创建模板耗时: {time.time()-start:.2f}秒")

start = time.time()
prompt = PromptTemplate(input_variables=["our_text", "wordsCount"], 
                            template = template)
print(f"创建PromptTemplate耗时: {time.time()-start:.2f}秒")

start = time.time()
final_prompt = prompt.format(our_text = "我喜欢旅行，我已经去过6个国家。我计划不久后再去几个国家。", 
              wordsCount = 3)
print(f"格式化提示词耗时: {time.time()-start:.2f}秒")

print("\n提示词内容:")
print(final_prompt)

start = time.time()
result = langchain_llm(final_prompt)
print(f"模型调用耗时: {time.time()-start:.2f}秒")

print("\n响应结果:")
print(result[:200] + "..." if len(result) > 200 else result)

print("\n===== 测试2: Few-Shot提示模板 =====")
start = time.time()
examples = [{'query': '什么是手机？',
             'answer': '手机是一种神奇的设备，可以装进口袋，就像一个迷你魔法游乐场。\
             它有游戏、视频和会说话的图片，但要小心，它也可能让大人变成屏幕时间的怪兽！'},
            {'query': '你的梦想是什么？',
             'answer': '我的梦想就像多彩的冒险，在那里我变成超级英雄，\
             拯救世界！我梦见欢笑声、冰淇淋派对，还有一只名叫Sparkles的宠物龙。'}]
print(f"创建示例耗时: {time.time()-start:.2f}秒")

start = time.time()
example_template = """
Question: {query}
Response: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)
print(f"创建示例模板耗时: {time.time()-start:.2f}秒")

start = time.time()
prefix = """你是一个5岁的小女孩，非常有趣、顽皮且可爱：
以下是一些例子：
"""

suffix = """
Question: {userInput}
Response: """
print(f"创建前缀和后缀耗时: {time.time()-start:.2f}秒")

start = time.time()
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n\n"
)
print(f"创建FewShotPromptTemplate耗时: {time.time()-start:.2f}秒")

start = time.time()
query = "房子是什么？"
print(f"创建查询耗时: {time.time()-start:.2f}秒")

start = time.time()
real_prompt = few_shot_prompt_template.format(userInput=query)
print(f"格式化Few-Shot提示词耗时: {time.time()-start:.2f}秒")

print("\nFew-Shot提示词内容:")
print(real_prompt)

start = time.time()
result = langchain_llm(real_prompt)
print(f"Few-Shot模型调用耗时: {time.time()-start:.2f}秒")

print("\nFew-Shot响应结果:")
print(result[:200] + "..." if len(result) > 200 else result)

print("\n===== 测试3: Chain调用 =====")
start = time.time()
chain = few_shot_prompt_template | langchain_llm
print(f"创建Chain耗时: {time.time()-start:.2f}秒")

start = time.time()
result = chain.invoke({"userInput": query})
print(f"Chain调用耗时: {time.time()-start:.2f}秒")

print("\nChain响应结果:")
print(result[:200] + "..." if len(result) > 200 else result)

print(f"\n总耗时: {time.time()-total_start:.2f}秒") 