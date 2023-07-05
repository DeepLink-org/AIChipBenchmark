import os
import sys
import json
import subprocess
import math
from command import parse_args
from utils import slurm_luanch_job


def get_abs_path(cwd, rpath):
    if rpath.startswith("/"):
        return rpath
    return os.path.join(cwd, rpath)


# returns true if model has not been successfully tested
def model_need_test(tested_infos, model_name, key):
    if not tested_infos:
        return True
    infos = tested_infos.get(model_name)
    if not infos:
        return True

    if not infos.get(key):
        return True

    elif isinstance(infos.get(key), float):
        if math.isnan(infos.get(key)):
            return True
    return False


openn_sbatch_cmd = "sbatch --exclusive --cpus-per-task 5 --mem=0 -o slurm_{jobname}.out -p {partition} -n {total_gpu} --ntasks-per-node {ngpu} --gres=gpu:{ngpu} \
 {sbatch_script} {model_config} {work_dir} {extra_args}"


# form a sbatch command and submmit the job
def run_model(
    resume,
    partition,
    total_gpu,
    gpu_per_node,
    sbatch_script,
    model_config,
    model_name,
    extra_args,
    cwd,
    is_perf,
    work_dir_base,
    tested_infos=None,
):
    # judge if bypass test
    dir_prefix = "perf" + str(total_gpu) if is_perf else "accu"
    if not model_need_test(tested_infos, model_name, dir_prefix):
        print("Omitting model test: " + model_name + " " + dir_prefix)
        return
    work_dir = os.path.join(get_abs_path(cwd, work_dir_base), model_name, dir_prefix)
    if os.path.exists(work_dir):
        dirs = os.listdir(work_dir)
        for file in dirs:
            if ".log" in file:
                os.system("rm -rf " + os.path.join(work_dir, file))

    # add resume-from flag if checkpoint exists and resume flag is set
    checkpoints = os.path.join(work_dir, "latest.pth")
    print(checkpoints)
    if resume and os.path.exists(checkpoints):
        extra_args = extra_args + " --resume-from " + checkpoints
        print(extra_args)

    cmd = openn_sbatch_cmd.format(
        partition=partition,
        total_gpu=total_gpu,
        ngpu=gpu_per_node,
        sbatch_script=sbatch_script,
        model_config=model_config,
        work_dir=work_dir,
        extra_args=extra_args,
        jobname=model_name + dir_prefix,
    )
    print(cmd)
    slurm_luanch_job(cmd, total_gpu / gpu_per_node)
    return work_dir


# the map contains the map from modelname to its class(cls,det,seg)
model_cls_map = dict()


def get_model_cls(cfgs, name):
    global model_cls_map
    if len(model_cls_map.keys()) == 0:
        for cls, v in cfgs.items():
            for model, model_cfs in v.items():
                model_cls_map[model] = cls
    if model_cls_map.get(name, None) is not None:
        return model_cls_map.get(name)


# return a list of model names
def get_model_list(cfgs):
    global model_cls_map
    model_list = []
    for cls, v in cfgs.items():
        for model, model_cfs in v.items():
            model_cls_map[model] = cls
            model_list.append(model)
    return model_list


# return the number of gpus to be used in accuracy test, according to model_config.json
def get_accu_gpu_num_from_cfg(cfg, model):
    cls = get_model_cls(cfg, model)
    cfg = cfg[cls]
    if cfg[model].get("accu_gpus", None) is not None:
        return cfg[model].get("accu_gpus")
    elif cfg["defaults"].get("accu_gpus", None) is not None:
        return cfg["defaults"].get("accu_gpus")
    else:
        return 8


# the args to be appended to command when running perf
perf_extra_args_map = {
    "cls": "--cfg-options runner.max_epochs=1 ",
    "det": "--cfg-options runner.max_epochs=1 ",
    "seg": "--cfg-options runner.max_iters=500 ",
}

model_cls_env_map = {
    "cls": "MMCLS_PATH",
    "det": "MMDET_PATH",
    "seg": "MMSEG_PATH",
}

log_interval_1 = "log_config.interval=1 "

if __name__ == "__main__":
    cwd = os.getcwd()
    base = os.path.join(cwd, "../")
    sbatch_script = os.path.join(cwd, "openmm_sbatch.sh")

    args = parse_args()

    # load config_file and restart_config
    config_file = args.config
    configs = {}
    with open(config_file, "r") as f:
        configs = json.load(f)

    tested_infos = None
    if args.restart_config:
        with open(args.restart_config, "r") as f:
            tested_infos = json.load(f)

    # get model_list from args or config(means all models will be tested)
    model_list = []
    if args.models is not None:
        model_list = args.models.split(",")
    elif args.cls is not None:
        for k, _ in configs[args.cls].items():
            if k != "defaults":
                model_list.append(k)
    else:
        model_list = get_model_list()

    print("models to test: ", model_list)

    perf_gpus = [1, 4, 8, 16]

    # iterate over model list, test each one
    for model_name in model_list:
        model_cls = get_model_cls(configs, model_name)

        # get the config file path needed by openmm train.py
        model_config = configs[model_cls][model_name]["cfg"]

        # goto the repo dir, like "cd mmclassification"
        base_dir = configs[model_cls]["defaults"]["base_dir"]
        base_dir = get_abs_path(base, base_dir)
        # get base_dir from ENV first
        if os.environ.get(model_cls_env_map[model_cls], None) is not None:
            base_dir = os.environ.get(model_cls_env_map[model_cls])

        model_config = get_abs_path(base_dir, model_config)

        os.chdir(base_dir)

        extra_args = perf_extra_args_map[model_cls] + log_interval_1

        # set the checkpoint interval bigger to save disk usage
        accu_extra_flags = " "
        if model_cls == "cls":
            accu_extra_flags = "--cfg-options checkpoint_config.interval=50"
        else:
            accu_extra_flags = "--cfg-options checkpoint_config.interval=20000"

        perf_exclude_list = configs["global"]["perf_excludes"]
        perf_exclude_flag = model_name in perf_exclude_list

        # test model accuracy
        if args.accu:
            total_gpu = get_accu_gpu_num_from_cfg(configs, model_name)
            log_dir = run_model(
                args.resume,
                args.partition,
                total_gpu,
                min(total_gpu, 8),
                sbatch_script,
                model_config,
                model_name,
                accu_extra_flags,
                base,
                False,
                args.work_dir,
                tested_infos,
            )

        if args.perf:
            # run perfs only if not excluded
            if perf_exclude_flag:
                print(model_name, " perf test skipped.")
            else:
                for ngpu in perf_gpus:
                    log_dir = run_model(
                        args.resume,
                        args.partition,
                        ngpu,
                        min(ngpu, 8),
                        sbatch_script,
                        model_config,
                        model_name,
                        extra_args,
                        base,
                        True,
                        args.work_dir,
                        tested_infos,
                    )
