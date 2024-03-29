from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(128, ), (256, ), (512, )],
        requires_grad=[True, False, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SEGMENTBASE2,
        url="https://gitlab.bj.sensetime.com/parrots.fit/segmentbase2/-/blob/master/segmentbase2/models/losses/knowledge_gap_loss.py#L27",  # noqa
        tags=[SampleTag.InputAware,\
              SampleTag.ViewAttribute, SampleTag.IfElseBranch,\
              SampleTag.Reduce, SampleTag.AdvancedIndexing]
    )


def gen_np_args(N):

    def gen_base(num):
        data = np.random.randn(num, 2, 1, 1) * 100
        data = data.astype(np.float32)
        return data

    logit = gen_base(N)
    label = gen_base(N)
    softlabel = gen_base(N)
    return [logit, label, softlabel]


register_sample(__name__, get_sample_config, gen_np_args)
