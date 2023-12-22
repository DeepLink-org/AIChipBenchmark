# import multiprocessing as mp
import argparse
import csv
import logging
import os
import os.path as osp
import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import List

import numpy as np
from pynvml import (NVMLError, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetMemoryInfo, nvmlDeviceGetName,
                    nvmlDeviceGetPowerState, nvmlDeviceGetTemperature,
                    nvmlInit, nvmlShutdown, nvmlSystemGetDriverVersion)
from tqdm import tqdm

from lmdeploy.tokenizer import Tokenizer
from lmdeploy.turbomind import TurboMind

import fire


def main(model_path: str, tp: int = 1):
    _start = time.perf_counter()
    tokenizer_model_path = osp.join(model_path, 'triton_models', 'tokenizer')
    tokenizer = Tokenizer(tokenizer_model_path)
    tm_model = TurboMind(model_path=model_path, tp=tp)
    _end = time.perf_counter()
    loading_model_elapsed_time = round(_end - _start, 2)
    # print(f'end TurboMind loading, elapsed time: {loading_model_elapsed_time}s')
    print(loading_model_elapsed_time)

if __name__ == '__main__':
    fire.Fire(main)
