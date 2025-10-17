import nemo_run as run

from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.utils.exp_manager import TimingCallback

import lightning.pytorch as pl
import torch
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_recipe(nodes: int = 1, gpus_per_node: int = 8):

    recipe = llm.qwen3_8b.pretrain_recipe(num_nodes=nodes, num_gpus_per_node=gpus_per_node,
                                            warmup_steps=100,
                                            max_steps=1000,
                                            val_check_interval=500)
    data_bak = recipe.data
    recipe.data=run.Config(
            PreTrainingDataModule,
            paths="datasets/processed2/arxiv_sample_text_document",
            seq_length=data_bak.seq_length,
            micro_batch_size=data_bak.micro_batch_size,
            global_batch_size=data_bak.global_batch_size,
            tokenizer=run.Config(AutoTokenizer, "./models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"),
            split='900,50,50',
            seed=2025,
    )
    recipe.trainer.callbacks=[run.Config(TimingCallback, log_tokens_per_sec = True)]

    # recipe.trainer.val_check_interval = 100
    return recipe

def local_executor_torchrun(nodes1: int = 1, devices: int = 8) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    # env_vars = {
    #     "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    #     "NCCL_NVLS_ENABLE": "0",
    #     "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
    #     "NVTE_ASYNC_AMAX_REDUCTION": "1",
    # }

    executor = run.LocalExecutor(nodes=nodes1, ntasks_per_node=devices)

    return executor

def run_pretraining():
    nodes=1
    gpus_per_node=8
    recipe = configure_recipe(nodes, gpus_per_node)
    executor = local_executor_torchrun(nodes1=recipe.trainer.num_nodes, devices=recipe.trainer.devices)
    run.run(recipe, executor=executor, name="qwen3_8b_pretraining")

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    pl.seed_everything(2025, workers=True, verbose=True)
    set_seed(2025)
    run_pretraining()