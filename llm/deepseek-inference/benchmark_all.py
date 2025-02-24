#!/usr/bin/env python3
import subprocess
import json
import csv
import os

# 测试配置
URL = "http://10.100.22.8:8222/v1/completions"
WORKER = 1
MODEL_PATH = "/data_share/model/DeepSeek-R1-INT8-MLU"
INPUT_NUM = 50

# 要测试的输入和输出长度组合
input_lengths = [256, 512]
output_lengths = [128, 512, 1024, 2048, 4096, 8192]

# CSV文件头
csv_headers = [
    'input_len', 'output_len', 'backend', 'clients', 'request_num',
    'Total QPS', 'Avg Input Length', 'Avg Output Length',
    'Total Throughput', 'Input Throughput', 'Output Throughput',
    'request_time_P1', 'request_time_P25', 'request_time_P50',
    'request_time_P75', 'request_time_P90', 'request_time_P95',
    'request_time_P99', 'request_time_P100',
    'first_token_time_P1', 'first_token_time_P25', 'first_token_time_P50',
    'first_token_time_P75', 'first_token_time_P90', 'first_token_time_P95',
    'first_token_time_P99', 'first_token_time_P100',
    'decode_token_time_P1', 'decode_token_time_P25', 'decode_token_time_P50',
    'decode_token_time_P75', 'decode_token_time_P90', 'decode_token_time_P95',
    'decode_token_time_P99', 'decode_token_time_P100'
]

# 创建CSV文件
with open('test_results.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()

    # 遍历所有输入输出长度组合
    for input_len in input_lengths:
        for output_len in output_lengths:
            print(f"\nTesting with input_len={input_len}, output_len={output_len}")
            
            # 构建命令
            cmd = [
                "python3", "/data_share/huangye/llm_profiler.py",
                "--url", URL,
                "--num_clients", str(WORKER),
                "--tokenizer_path", MODEL_PATH,
                "--input_len", str(input_len),
                "--output_len", str(output_len),
                "--input_num", str(INPUT_NUM),
                "--trust_remote_code"
            ]

            log_file = f"test_log_input{input_len}_output{output_len}.log"
            try:
                # 执行命令并将输出写入日志文件
                with open(log_file, 'w') as f:
                    result = subprocess.run(cmd, stdout=f, stderr=f, text=True, check=True)
                # import pdb;pdb.set_trace()
                # 从日志文件中读取并解析JSON结果
                with open(log_file, 'r') as f:
                    # 逐行读取文件内容到列表中
                    lines = f.readlines()
                    # 去除每行的空白字符并过滤掉空行
                    lines = [line.strip() for line in lines if line.strip()]
                    
                    # 从后向前查找有效的JSON内容
                    json_found = False
                    for line in reversed(lines):
                        try:
                            data = eval(line)
                            json_found = True
                            print(f"Found valid JSON data: {line[:100]}...")
                            break
                        except json.JSONDecodeError:
                            continue
                    if not json_found:
                        print(f"Warning: Last {min(5, len(lines))} lines of log file:")
                        for line in lines[-5:]:
                            print(f"  {line[:100]}...")
                        raise json.JSONDecodeError("No valid JSON found in log file", "", 0)    

                # 准备CSV行数据
                row = {
                    'input_len': input_len,
                    'output_len': output_len,
                    'backend': data.get('backend', ''),
                    'clients': data['clients'],
                    'request_num': data['request_num'],
                    'Total QPS': data['Total QPS'],
                    'Avg Input Length': data['Avg Input Length'],
                    'Avg Output Length': data['Avg Output Length'],
                    'Total Throughput': data['Total Throughput'],
                    'Input Throughput': data['Input Throughput'],
                    'Output Throughput': data['Output Throughput']
                }

                # 添加请求时间分位数据
                for p in ['P1', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99', 'P100']:
                    row[f'request_time_{p}'] = data['request_time'][p]
                    row[f'first_token_time_{p}'] = data['first_token_time_dict'][p]
                    row[f'decode_token_time_{p}'] = data['decode_token_time_dict'][p]

                # 写入CSV
                writer.writerow(row)
                print(f"Results written to CSV for input_len={input_len}, output_len={output_len}")

            except subprocess.CalledProcessError as e:
                print(f"Error running test: {e}")
                print(f"Error output: {e.stderr}")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON output: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

print("\nAll tests completed. Results saved in test_results.csv")
