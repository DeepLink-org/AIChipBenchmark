import os
import sys
import re
import json


# parse a log file, return the time
def get_time(lines):
    for l in lines:
        if re.match("^[0-9].+[0-9]", l):
            try:
                nums = l.strip().split(" ")
                nbytes = int(nums[0])
                timeus = round(float(nums[-1]) / 1000, 3)
                return (int(nbytes / 4), timeus)
            except:
                print("Failed to parse line: " + l)


# read a log file and write result back to infos
def parse_file(log, infos, commtype):
    with open(log, "r") as f:

        ngpu = log.split("_")[-2].split(".")[0]
        ret = get_time(f.readlines())
        if not ret:
            print("failed to parse: " + log)
            return
        comm_offload = str(ret[0])
        if not infos.get(ngpu):
            infos[ngpu] = {}
        if not infos[ngpu].get(comm_offload):
            infos[ngpu][comm_offload] = {}
        if not infos[ngpu][comm_offload].get("latency"):
            infos[ngpu][comm_offload]["latency"] = ret[1]
        else:
            # already has a value
            ori = infos[ngpu][comm_offload]["latency"]
            if abs(ori - ret[1]) / ori > 0.2:
                infos[ngpu][comm_offload]["latency"] = min(ori, ret[1])
            else:
                infos[ngpu][comm_offload]["latency"] = round((ori + ret[1]) / 2, 3)
        # update bandwidth
        ngpu_int = int(ngpu)
        if commtype == "all-gather":
            comm_bytes = ret[0] * 4 * ngpu_int * (ngpu_int - 1) / ngpu_int
        elif commtype == "all-reduce":
            comm_bytes = ret[0] * 4 * 2 * (ngpu_int - 1) / ngpu_int
        else:
            print("Not Supported Comm Ops, set comm_bytes to 0")
            comm_bytes = 0
        bandwidth = round(comm_bytes / (infos[ngpu][comm_offload]["latency"] * 1e6), 2)
        infos[ngpu][comm_offload]["bandwidth"] = bandwidth


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: parse_comm_result log_dir output_json")
    log_dir = sys.argv[1]
    outjson = sys.argv[2]
    commtype = sys.argv[3]
    infos = {}
    # read existing infos if files exits
    if os.path.exists(outjson):
        print("Updating " + outjson)
        with open(outjson, "r") as f:
            try:
                infos = json.load(f)
            except:
                pass
    files = os.listdir(log_dir)
    for file in files:
        fp = os.path.join(log_dir, file)
        parse_file(fp, infos, commtype)
    with open(outjson, "w") as f:
        json.dump(infos, f)

