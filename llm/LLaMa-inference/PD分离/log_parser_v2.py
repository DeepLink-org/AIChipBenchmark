'''
安装依赖：
pip install pandas openpyxl

使用步骤：
1. 确保所有日志文件存放在指定目录（默认：./logs）
2. 运行脚本：python log_parser_v2.py 即生成报告文件 benchmark_report.xlsx 到当前目录下
'''

import os
import re
import pandas as pd

def parse_log_file(file_path):
    filename = os.path.basename(file_path)
    
    pattern = r"input_len-(\d+).*?qps-(\d+)"
    match = re.search(pattern, filename)
    if not match:
        return None
    
    input_len = int(match.group(1))
    qps = int(match.group(2))
    
    metrics = {
        "qps": qps,
        "input len": input_len,
        "Median TTFT (ms)": None,
        "Median TPOT (ms)": None
    }

    with open(file_path, 'r') as f:
        content = f.read()
        
        ttft_match = re.search(r"Median TTFT \(ms\):\s+([\d.]+)", content)
        if ttft_match:
            metrics["Median TTFT (ms)"] = float(ttft_match.group(1))
        
        tpot_match = re.search(r"Median TPOT \(ms\):\s+([\d.]+)", content)
        if tpot_match:
            metrics["Median TPOT (ms)"] = float(tpot_match.group(1))

    return metrics

def generate_report(logs_dir, output_file):
    data = []
    
    for filename in os.listdir(logs_dir):
        if filename.endswith(".log"):
            file_path = os.path.join(logs_dir, filename)
            result = parse_log_file(file_path)
            if result:
                data.append(result)
    

    df = pd.DataFrame(data)
    df = df.sort_values(by=["qps", "input len"])
    
    column_order = ["qps", "input len", "Median TTFT (ms)", "Median TPOT (ms)"]
    df = df[column_order]
    
    df.to_excel(output_file, index=False)
    print(f"报告已生成：{output_file}")

if __name__ == "__main__":
    # 配置参数
    logs_dir = "./logs"          # 日志目录路径
    output_file = "benchmark_report.xlsx"  # 输出文件名
    
    generate_report(logs_dir, output_file)