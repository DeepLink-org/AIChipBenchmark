import nemo_run as run
from datetime import timedelta
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

    recipe = llm.llama3_8b.pretrain_recipe(num_nodes=nodes, num_gpus_per_node=gpus_per_node)

    # 修改 trainer 参数
    recipe.trainer.max_steps = 100
    recipe.trainer.val_check_interval = 200
    recipe.trainer.strategy.tensor_model_parallel_size = 2
    recipe.trainer.strategy.pipeline_model_parallel_size = 1
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None
    recipe.trainer.strategy.use_te_rng_tracker = True


    recipe.data=run.Config(
            PreTrainingDataModule,
            paths="./llama3-8B/arxiv_sample_text_document",
            seq_length=8192,
            micro_batch_size=1,
            global_batch_size=128,
            tokenizer=run.Config(AutoTokenizer, "./models--llama3--llama3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"),
            split='900,50,50',
            seed=2025,
    )
    
    # 启用性能优化
    recipe.model.config.enable_cuda_graph = True
    recipe.model.config.cross_entropy_fusion_impl = "te"
    recipe.model.config.gradient_accumulation_fusion = False
    recipe.model.config.masked_softmax_fusion = True
    recipe.model.config.cross_entropy_loss_fusion = True
    recipe.model.config.apply_rope_fusion = True
    recipe.model.config.bias_dropout_fusion = True
    recipe.model.config.bias_activation_fusion = True
    # 完全关闭保存ckpt，减少io开销
    recipe.log.ckpt = None

    # 设置 FSDP 和 DDP
    recipe.trainer.strategy.fsdp = "megatron"
    recipe.trainer.strategy.ddp = run.Config(
        DistributedDataParallelConfig,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        average_in_collective=True,
        data_parallel_sharding_strategy="optim_grads_params",
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
    nodes=1
    gpus_per_node=8
    recipe = configure_recipe(nodes, gpus_per_node)
    executor = local_executor_torchrun(nodes1=recipe.trainer.num_nodes, devices=recipe.trainer.devices)
    run.run(recipe, executor=executor, name="llama3_8b_pretraining")

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    pl.seed_everything(2025, workers=True, verbose=True)
    set_seed(2025)
    run_pretraining()