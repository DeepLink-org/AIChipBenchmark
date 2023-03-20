import torch
import os
import logging

logger = logging.getLogger("ResultDiff")
LOGLEVEL = os.environ.get("PYLOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)


def to_list(obj):
    if isinstance(obj, list):
        return obj
    else:
        return [
            obj,
        ]


def output_to_list(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return obj
    else:
        return [
            obj,
        ]


@torch.no_grad()
def tensor_diff(t1, t2, abs_thresh=1e-8, relative_thresh=1e-5, total_thresh=0.001):
    """
    Description: Compares two torch tensor t1 and t2 element-wise with thresh.
                 Similar to allclose() but different.
    Param:
        t1 & t2: input tensor to compate
        abs_thresh: absolute error threshold
        relative_thresh: relative error threshold
        total_thresh: the percentage of elements exceeds the abs_thresh
            or relative_thresh should be lower than total_thresh
    return: bool
        where the inputs passed the diff threshold
    """
    t1 = t1.detach().float()
    t2 = t2.detach().float()
    if torch.numel(t1) != torch.numel(t2):
        return False
    if t1.dtype == torch.bool:
        return tensor_diff_bool(t1, t2, total_thresh)
    abs_diff = torch.abs(t1 - t2)
    # reference: numpy.allclose
    all_err = abs_diff > (abs_thresh + relative_thresh * torch.abs(t2))

    return (all_err.sum() / torch.numel(t2)) < total_thresh


def tensor_diff_bool(t1, t2, total_thresh=0.001):
    """
    for bool tensors, only count the non-equal elements
    """
    assert t1.dtype == torch.bool
    assert t2.dtype == torch.bool
    xor = torch.logical_xor(t1, t2)
    return (xor.sum() / torch.numel(t2)) < total_thresh


def result_diff(t1, t2, abs_thresh=1e-8, relative_thresh=1e-5, total_thresh=0.001, msg=""):
    """
    Description: Compares two torch tensor t1 and t2. Dispatches to different cases.
                 This function will try different thresholds on test failure.
    """
    if isinstance(t1, torch.Tensor):
        t1 = t1.cpu()
        t2 = t2.cpu()
        rst = tensor_diff(t1, t2, abs_thresh, relative_thresh, total_thresh)
        if not rst:
            logger.warning(f"{msg} failed to pass relative_thresh={relative_thresh} abs_thresh={abs_thresh}")
            if tensor_diff(t1, t2, abs_thresh=1e-8, relative_thresh=1e-5):
                logger.info(msg + f" passed test on relative_thresh={1e-5} abs_thresh={1e-8}")
            elif tensor_diff(t1, t2, abs_thresh=1e-5, relative_thresh=1e-4):
                logger.info(msg + f" passed test on relative_thresh={1e-4} abs_thresh={1e-5}")
            elif tensor_diff(t1, t2, abs_thresh=1e-4, relative_thresh=1e-3):
                logger.info(msg + f" passed test on relative_thresh={1e-3} abs_thresh={1e-4}")
            elif tensor_diff(t1, t2, abs_thresh=1e-3, relative_thresh=1e-2):
                logger.info(msg + f" passed test on relative_thresh={1e-2} abs_thresh={1e-3}")
            else:
                logger.warning(msg + " Failed to pass any test")
        else:
            logger.info(msg + f"passed test on relative_thresh={relative_thresh} abs_thresh={abs_thresh}")
        return rst
    elif isinstance(t1, bool):
        return t1 == t2
    elif t1 is None and t2 is None:
        logger.warning("got None")
    else:
        logger.warning("Failed to diff tensors, one tensor might be empty")
        return False


def fix_rand(rank=0):
    import numpy as np
    import random

    seed = 2222 + rank
    logger.info(f"Setting random seed to {seed}")

    # PyTorch random number generator (for cpu and cuda)
    torch.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    # python random
    random.seed(seed)

    # numpy RNG
    np.random.seed(seed)

    # cuda benchmarking
    torch.backends.cudnn.benchmark = False

    # deterministic algos
    torch.use_deterministic_algorithms(True)

    # cudnn conv deterministic
    torch.backends.cudnn.deterministic = True

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.random.manual_seed(seed)


# try to use deterministic algos first
def try_deterministic(func):
    def warpper(*args):
        try:
            torch.use_deterministic_algorithms(True)
            func(*args)
        except:
            logger.info(
                "Got exception with use_deterministic_algorithms=True, retrying use_deterministic_algorithms=False"
            )
            torch.use_deterministic_algorithms(False)
            func(*args)

    return warpper


# do not use low precision options on CUDA
def turn_off_low_precision():
    logger.info("disable allow tf32")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.__version__ == "1.12.0":
        logger.info("diable torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction")
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
