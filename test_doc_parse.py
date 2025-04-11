import os
import sys
import time
from doc_parse import read_and_process_excel

def test_read_and_process_excel():
    """测试Excel文件处理功能"""
    print("===== 测试 Excel 处理功能 =====")
    
    # 检查是否有测试用的Excel文件
    excel_path = "./data/zhidu_detail.xlsx"
    
    if not os.path.exists(excel_path):
        print(f"错误: 未找到测试用的Excel文件: {excel_path}")
        return False
    
    print(f"使用测试文件: {excel_path}")
    
    try:
        start_time = time.time()
        data = read_and_process_excel(excel_path)
        end_time = time.time()
        
        print(f"成功读取Excel文件")
        print(f"数据行数: {len(data)}")
        if len(data) > 0:
            print(f"第一行数据: {data[0][:5]}...")
        print(f"耗时: {end_time - start_time:.2f}秒")
        print("Excel处理功能测试成功!\n")
        return True
    except Exception as e:
        print(f"Excel处理功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试doc_parse.py的Excel处理功能...\n")
    
    # 测试Excel处理功能
    excel_success = test_read_and_process_excel()
    
    if excel_success:
        print("Excel处理功能工作正常")
        sys.exit(0)
    else:
        print("Excel处理功能测试失败")
        sys.exit(1) 