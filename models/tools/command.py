import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Test models")
    parser.add_argument("--work-dir", default="work_tmp", help="the dir to save logs and models")
    parser.add_argument("--models", help="the name of models to test")
    parser.add_argument("--partition", help="the partition of slurm")
    parser.add_argument("--config", help="the model config file")
    parser.add_argument("--restart-config", help="model training result json to continue with")
    parser.add_argument("--cls", help="the class of models to test, e.g. cls/det/seg")
    parser.add_argument("--perf", action="store_true", help="whether not to evaluate the training performance")
    parser.add_argument("--accu", action="store_true", help="whether not to evaluate the training accuracy")
    parser.add_argument("--resume", action="store_true", help="whether not to resume from the checkpoint file")

    # for parsing results
    parser.add_argument("--model_log", help="the directory of a model log")
    parser.add_argument("--log_root", help="the directory of all models log")
    parser.add_argument("--output", default="output_report", help="the directory of output jsons")

    args = parser.parse_args()
    return args
