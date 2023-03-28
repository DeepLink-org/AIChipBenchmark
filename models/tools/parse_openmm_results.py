import os
import sys
import json
import numpy as np
from collections import defaultdict
import math

from command import parse_args


def load_json_log(json_log):
    """load and convert json_logs to log_dicts.

    Args:
        json_log (str): The path of the json log file.

    Returns:
        dict[int, dict[str, list]]:
            Key is the epoch, value is a sub dict. The keys in each sub dict
            are different metrics, e.g. memory, bbox_mAP, and the value is a
            list of corresponding values in all iterations in this epoch.

            .. code-block:: python

                # An example output
                {
                    1: {'iter': [100, 200, 300], 'loss': [6.94, 6.73, 6.53]},
                    2: {'iter': [100, 200, 300], 'loss': [6.33, 6.20, 6.07]},
                    ...
                }
    """
    log_dict = dict()
    with open(json_log, "r") as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # skip lines without `epoch` field
            if "epoch" not in log:
                continue
            epoch = log.pop("epoch")
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict


# modified from mmclassification
def cal_train_time(log_dict):
    all_times = []
    for epoch in log_dict.keys():
        # skip the first three iters and count only the first epoch
        if epoch == 1:
            all_times.append(np.array(log_dict[epoch]["time"][3:]))
    all_times = np.array(all_times)
    epoch_ave_time = all_times.mean(-1)
    slowest_epoch = epoch_ave_time.argmax()
    fastest_epoch = epoch_ave_time.argmin()
    std_over_epoch = epoch_ave_time.std()
    return np.mean(all_times)


# get the key=kw value from dict
def get_accu_results(log_dict, kw):
    try:
        ind = list(log_dict.keys())[-1]
        return log_dict[ind][kw][0]
    except:
        print("get accu result failed!")
        return -1


# get the (samples_per_gpu ,num_gpu) info from log
def get_batchsize_from_log(log):
    sample_per_gpu = 0
    N_GPU = 0
    with open(log) as f:
        lines = f.readlines()
        for l in lines:
            # get samples_per_gpu
            kw = "samples_per_gpu="
            pos = l.find(kw)
            if pos != -1:
                key_info = l[pos + len(kw) : pos + len(kw) + 4]
                sample_per_gpu = int(key_info.split(",")[0])
            # get num_gpu
            kw = "gpu_ids = range(0, "
            pos = l.find(kw)
            if pos != -1:
                key_info = l[pos + len(kw) : pos + len(kw) + 2]
                N_GPU = int(key_info.split(")")[0])
    return sample_per_gpu, N_GPU


# parse a single log.json, and return perf or accuracy
def parse_log(log_json, configs=None, perf=True, model_name=None):
    infos = load_json_log(log_json)
    # calculate IPS from time and batchsize
    if perf:
        mean_iter_time = None
        sample_per_gpu = 1
        nGPU = 1
        mean_iter_time = cal_train_time(infos)
        sample_per_gpu, nGPU = get_batchsize_from_log(log_json)
        IPS = round(nGPU * sample_per_gpu / mean_iter_time, 2)
        if math.isnan(IPS):
            IPS = 0
        return {"IPS": IPS, "ngpu": nGPU, "sample_per_gpu": sample_per_gpu}

    assert model_name is not None

    # get accuracy result from log, need the accuracy keyword to lookup
    accu_info = dict()
    kw = configs[model_name]
    ret = {}
    if isinstance(kw, list):
        for key in kw:
            ret[key] = get_accu_results(infos, key)
        return ret
    else:
        accu = get_accu_results(infos, kw)
        return {kw: accu} if accu != -1 else {}


# traversal a single models output folder and get the perf & accuracy statistics
def traversal_model_folder(base, model_info_map):
    if base.endswith("/"):
        base = base[:-1]
    model_name = base.split("/")[-1]
    base_list = os.listdir(base)
    model_infos = {}
    # traversal all files in the log dir
    for l in base_list:
        if l == "accu":
            for file in os.listdir(os.path.join(base, l)):
                if file.endswith(".log.json"):
                    result = parse_log(os.path.join(base, l, file), model_info_map, False, model_name)
                    model_infos[l] = result
                    if not result:
                        print("parse failed: " + os.path.join(base, l, file))
        elif "perf" in l:
            ngpu = int(l[4:])
            for file in os.listdir(os.path.join(base, l)):
                if file.endswith(".log.json"):
                    result = parse_log(os.path.join(base, l, file), model_info_map, True, model_name)
                    model_infos[l] = result["IPS"]

                    if result["ngpu"] != ngpu:
                        print("Error! Get different num of gpus!")
    return {model_name: model_infos}


# traversal the root output folder and get the perf & accuracy statistics for all models
def traversal_folder(log_root, model_info_map):
    ret = dict()
    fds = os.listdir(log_root)

    for fd in fds:
        if os.path.isdir(os.path.join(log_root, fd)):
            # parse a single model
            result = traversal_model_folder(os.path.join(log_root, fd), model_info_map)
            if not ret:
                ret = result
            else:
                ret.update(result)
    return ret


if __name__ == "__main__":
    args = parse_args()

    cwd = os.getcwd()
    base = os.path.join(cwd, "../")
    config_file = args.config

    # first build a map from modelname to its class
    model_info_map = dict()
    with open(config_file) as f:
        jsons = json.load(f)
        for cls, val in jsons.items():
            if cls not in ["cls", "det", "seg"]:
                continue
            # get the default accuracy keyword for current model
            cls_default = jsons[cls]["defaults"]["accu_key_value"]
            for model, infos in val.items():
                if infos.get("accu_key_value", None) is not None:
                    model_info_map[model] = infos.get("accu_key_value")
                else:
                    model_info_map[model] = cls_default

    ret = None
    if args.model_log:
        ret = traversal_model_folder(args.model_log, model_info_map)
    if args.log_root:
        ret = traversal_folder(args.log_root, model_info_map)

    print(ret)

    results = {}
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            try:
                results = json.load(f)
            except:
                pass

    results.update(ret)

    with open(args.output, "w") as f:
        json.dump(results, f)
