import nemo_run as run

from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.utils.exp_manager import TimingCallback
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
from nemo.lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from megatron.core.distributed import DistributedDataParallelConfig

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

    recipe = llm.llama3_70b.pretrain_recipe(num_nodes=nodes, num_gpus_per_node=gpus_per_node)

    # 修改 trainer 参数
    recipe.trainer.max_steps = 100
    recipe.trainer.val_check_interval = 200
    recipe.trainer.strategy.tensor_model_parallel_size = 4  
    recipe.trainer.strategy.pipeline_model_parallel_size = 4
    recipe.trainer.strategy.context_parallel_size = 2
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = 5
    recipe.trainer.strategy.use_te_rng_tracker = True
    recipe.trainer.strategy.ckpt_async_save=False

    recipe.data=run.Config(
            PreTrainingDataModule,
            paths="./llama3-70B/arxiv_sample_text_document",
            seq_length=8192,
            micro_batch_size=2,
            global_batch_size=64,
            tokenizer=run.Config(AutoTokenizer, "./models--llama3--llama3-70B/snapshots/7cde9a27957f27ce5677b1f838ccaeeb69acc8d0"),
            split='900,50,50',
            seed=2025,
    )
    

    # 启用性能优化
    recipe.model.config.enable_cuda_graph = False
    recipe.model.config.cross_entropy_fusion_impl = "te"
    recipe.model.config.gradient_accumulation_fusion = False
    recipe.model.config.masked_softmax_fusion = True
    recipe.model.config.cross_entropy_loss_fusion = True
    recipe.model.config.apply_rope_fusion = True
    recipe.model.config.bias_dropout_fusion = True
    recipe.model.config.bias_activation_fusion = True

    # 设置 DsDP
    recipe.trainer.strategy.ddp = run.Config(
        DistributedDataParallelConfig,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        average_in_collective=True,
    )
    
    recipe.trainer.callbacks = []
    recipe.trainer.callbacks=[run.Config(TimingCallback, log_tokens_per_sec = True)]
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
    nodes=4
    gpus_per_node=8
    recipe = configure_recipe(nodes, gpus_per_node)
    executor = local_executor_torchrun(nodes1=recipe.trainer.num_nodes, devices=recipe.trainer.devices)
    run.run(recipe, executor=executor, name="llama3_70b_pretraining")

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    pl.seed_everything(2025, workers=True, verbose=True)
    set_seed(2025)
    run_pretraining()