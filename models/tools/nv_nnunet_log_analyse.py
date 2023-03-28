import json
import sys
import os

import argparse

from utils import merge_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse nnunet log json")
    parser.add_argument("--log", help="the log json file")
    parser.add_argument("--param", help="the param json file")
    parser.add_argument("--output", default="nnunet_out.json", help="the output json file")
    parser.add_argument("--perf", action="store_true", help="whether not to evaluate the training performance")
    parser.add_argument("--accu", action="store_true", help="whether not to evaluate the training accuracy")
    args = parser.parse_args()
    return args


def get_accu(log_dict):
    return log_dict["data"]


if __name__ == "__main__":
    args = parse_args()
    log_dict = None
    result = {"nnunet": {}}
    with open(args.log, "r") as f:
        lines = f.readlines()
        # get only the last line
        log_dict = json.loads(lines[-1][5:])
    if args.accu:
        ret = get_accu(log_dict)
        result["nnunet"]["accu"] = ret
    # get perf result
    if args.perf:
        params = None
        with open(args.param, "r") as f:
            params = json.load(f)
        nnode = params["nodes"]
        ngpu = params["gpus"] * nnode
        result["nnunet"]["perf" + str(ngpu)] = log_dict["data"]["throughput_train"]

    # if output json already exists, then read it out and merge with current result
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            try:
                outputs = json.load(f)
                result = merge_dict(outputs, result)
            except:
                pass
    # dump result to output
    with open(args.output, "w") as f:
        outputs = json.dump(result, f)
