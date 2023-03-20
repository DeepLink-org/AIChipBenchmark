from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)

import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(4000, 4)],
        requires_grad=[False] * 2,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.POD,
        url="https://gitlab.bj.sensetime.com/platform/ParrotsDL/pytorch-object-detection/-/blob/pt/v3.1.0/pod/models/heads/utils/matcher.py#L147",  # noqa
        tags=[SampleTag.ViewAttribute, \
              SampleTag.IfElseBranch]
    )


def gen_np_args(M, N):
    boxes = np.random.randn(M, N)
    boxes = boxes.astype(np.float32)
    gt = np.random.randn(M, 5)
    gt = gt.astype(np.float32)
    return [boxes, gt]


register_sample(__name__, get_sample_config, gen_np_args)
