import argparse
import requests
import json
import time
import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

model_name_in_post = []


def get_random_input_data(input_len, input_num, tokenizer_path):
    prompts = []
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    input_lens = []
    for _ in range(input_num):
        candidate_ids = [
            random.randint(10, vocab_size - 1)
            for _ in range(input_len)
        ]
        candidate_prompt = tokenizer.decode(candidate_ids)
        prompts.append(candidate_prompt)
        input_lens.append(input_len)
    return prompts, input_lens


def get_output_length(input_num, output_len):
    min_len, max_len = 2, output_len * 2
    mean = (min_len + max_len) * 0.5
    std = mean #(max_len - mean) / 3.0 # 3std准则
    output_lens = []
    for _ in range(input_num):
        cur_len = random.gauss(mean, std)
        cur_len = round(cur_len)
        if cur_len < min_len:
            cur_len = min_len
        elif cur_len > max_len:
            cur_len = max_len
        output_lens.append(cur_len)
    return output_lens 


def post_data_decorator(func, request_gen):
    def wrapper(url, input, max_new_tokens):
        request_data, _ = request_gen(input, max_new_tokens)
        result = func(url, request_data)
        return result
    return wrapper


def post_stream_vllm(url, request_data):
    headers = {'Content-Type': 'application/json'}
    used_time = []
    start_time = time.time()
    last_time = start_time
    response = requests.post(url, headers=headers, data=json.dumps(request_data), stream=True)
    assert response.status_code == 200
    generated_text=""
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")[6:]    # remove "data: "
            if line == "[DONE]":
                continue
            data = json.loads(line) 
            if data["choices"][0]["text"]:
                generated_text += data["choices"][0]["text"]
            current_time = time.time()
            elapsed_time = current_time - last_time
            used_time.append(elapsed_time)
            last_time = current_time
    return used_time


def gen_vllm_request(inputs, max_new_tokens):
    prompt = inputs
    request_data = {
        "model": model_name_in_post[0],
        "prompt": inputs,
        "n": 1,
        'ignore_eos': True,
        'max_tokens': max_new_tokens, 
        "stream": True,
    }

    return request_data, prompt


def main(args):
    model_name_in_post.append(args.tokenizer_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    url = args.url
    percentiles = [1, 25, 50, 75, 90, 95, 99, 100]

    model_name_in_post.append(args.tokenizer_path)
    post_stream = post_data_decorator(post_stream_vllm, gen_vllm_request)
    prompts, input_lens = get_random_input_data(args.input_len, args.input_num, args.tokenizer_path)
    max_new_tokens = get_output_length(len(prompts), args.output_len)

    dump_dict = {}
    dump_dict["backend"] = args.server_api
    dump_dict["clients"] = args.num_clients

    start_time = time.time()
    if args.post_mode == "stream":
        with ThreadPoolExecutor(max_workers=args.num_clients) as executor:
            results = list(tqdm(executor.map(lambda p: post_stream(url, p[0], p[1]), zip(prompts, max_new_tokens)), total=len(prompts), desc="Running tests"))
        end_time = time.time()
        first_token_time = []
        decode_token_time = []
        request_time = []
        final_output_lens = []
        valid_num = 0
        for result in results:
            if len(result) > 1: # 统计至少decode出两个token的数据
                first_token_time.append(result[0])
                decode_token_time.append(sum(result[1:]) / len(result[1:]))
                request_time.append(sum(result))
                final_output_lens.append(len(result))
                valid_num += 1
        print(f"\n\nvalid num = {valid_num}; all data num = {len(results)}; valid ratio = {valid_num * 1.0 / len(results)}\n")
        print(f"Total QPS: {valid_num / (end_time - start_time)}")
        print(f"Avg Input Length: {sum(input_lens) / len(input_lens)}")
        print(f"Avg Output Length: {sum(final_output_lens) / len(final_output_lens)}")
        print(f"Total Throughput: {(sum(input_lens) + sum(final_output_lens)) / (end_time - start_time)} token/s")
        print(f"Input Throughput: {sum(input_lens) / (end_time - start_time)} token/s")
        print(f"Output Throughput: {sum(final_output_lens) / (end_time - start_time)} token/s")
        print("-" * 10)
        dump_dict["request_num"] = valid_num
        dump_dict["Total QPS"] = valid_num / (end_time - start_time)
        dump_dict["Avg Input Length"] = sum(input_lens) / len(input_lens)
        dump_dict["Avg Output Length"] = sum(final_output_lens) / len(final_output_lens)
        dump_dict["Total Throughput"] = (sum(input_lens) + sum(final_output_lens)) / (end_time - start_time)
        dump_dict["Input Throughput"] = sum(input_lens) / (end_time - start_time)
        dump_dict["Output Throughput"] = sum(final_output_lens) / (end_time - start_time)

        values = np.percentile(request_time, percentiles)
        request_time_dict = {}
        for percentile, value in zip(percentiles, values):
            print(f"request_time P{percentile}: {value:.6f}s")
            request_time_dict[f"P{percentile}"] = value
        dump_dict["request_time"] = request_time_dict 
        print("-" * 10)
        
        first_token_time_dict = {}
        values = np.percentile(first_token_time, percentiles)
        for percentile, value in zip(percentiles, values):
            print(f"first_token_time  P{percentile}: {value:.6f}s")
            first_token_time_dict[f"P{percentile}"] = value
        dump_dict["first_token_time_dict"] = first_token_time_dict
        print("-" * 10)

        decode_token_time_dict = {}
        values = np.percentile(decode_token_time, percentiles)
        for percentile, value in zip(percentiles, values):
            print(f"decode_token_time  P{percentile}: {value * 1000:.6f}ms")
            decode_token_time_dict[f"P{percentile}"] = value * 1000
        dump_dict["decode_token_time_dict"] = decode_token_time_dict
        print(dump_dict)
    else:
        raise Exception(f'Not support {args.post_mode} post_mode.')
    
    if args.dump_file:
        with open(args.dump_file, 'w') as json_file:
            json.dump(dump_dict, json_file, indent=4)
        print(f"Results have been written to {args.dump_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--url', type=str, default="")
    parser.add_argument('--num_clients', type=int, default=0)
    parser.add_argument('--post_mode', type=str, choices=["", "stream", "nostream"], default="stream")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tokenizer_path', type=str, default="")
    parser.add_argument('--data_mode', type=str, default="")
    parser.add_argument('--input_len', type=int, default=0)
    parser.add_argument('--input_num', type=int, default=0)
    parser.add_argument('--output_len', type=int, default=0)
    parser.add_argument('--server_api', type=str, default="")
    parser.add_argument('--dump_file', type=str, default="")
    parser.add_argument('--trust_remote_code', action='store_true', default=False)

    args = parser.parse_args()

    main(args)

# python3 llm_profile.py \
#     --url ${URL} \
#     --num_clients ${WORKER} \
#     --tokenizer_path ${MODEL_PATH} \
#     --input_len 2048 \
#     --output_len 128 \
#     --input_num 200 \
#     --trust_remote_code