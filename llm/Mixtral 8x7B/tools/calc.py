# calc.py
import re
import sys

def calculate_average_metrics(log_file_path, start_iter, end_iter):
    """
    计算指定迭代次数范围内某个度量的平均值。

    :param log_file_path: 日志文件的路径
    :param start_iter: 开始迭代次数
    :param end_iter: 结束迭代次数
    :return: 指定迭代次数范围内的平均度量值
    """
    total_metrics = 0
    count = 0
    with open(log_file_path, 'r') as file:
        lines = file.readlines()  # 读取所有行到列表
        # 遍历所有行
        for i in range(len(lines)):
            line = lines[i]
            tokens_per_sec1 = "tokens_per_sec"
            metrics_pattern = fr'{tokens_per_sec1}: (\d+\.\d+)'
            metrics_match = re.search(metrics_pattern, line)
            if metrics_match:
                iter_match = re.search(r'Iter\(train\)\s\[(\d+)/', line)
                if iter_match:
                    iteration_number = int(iter_match.group(1))
                    if start_iter <= iteration_number <= end_iter:
                        elapsed_metrics = float(metrics_match.group(1))
                        total_metrics += elapsed_metrics
                        count += 1
    average_ms = total_metrics / count if count else 0
    return f"{average_ms:.2f}"

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python calc.py <log_file_path> <start_iter> <end_iter>")
        sys.exit(1)

    log_file_path = sys.argv[1]
    start_iter = int(sys.argv[2])
    end_iter = int(sys.argv[3])

    average_metrics = calculate_average_metrics(log_file_path, start_iter, end_iter)
    print(average_metrics)