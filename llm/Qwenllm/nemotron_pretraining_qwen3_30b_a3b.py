import nemo_run as run
from datetime import timedelta
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.utils.exp_manager import TimingCallback
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
from nemo.lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from megatron.core.distributed import DistributedDataParallelConfig
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback

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

    recipe = llm.qwen3_30b_a3b.pretrain_recipe(num_nodes=nodes, num_gpus_per_node=gpus_per_node,
                                         tensor_parallelism=1,
                                         pipeline_parallelism=1,
                                         context_parallelism=1,
                                         virtual_pipeline_parallelism=None,
                                         global_batch_size=32,
                                         micro_batch_size=2,
                                            warmup_steps=10,
                                            max_steps=100,
                                            val_check_interval=100)
    data_bak = recipe.data
    recipe.data=run.Config(
            PreTrainingDataModule,
            paths="./datasets_precessed/qwen3_30b_a3b/arxiv_sample_text_document",
            seq_length=4096,
            micro_batch_size=data_bak.micro_batch_size,
            global_batch_size=data_bak.global_batch_size,
            tokenizer=run.Config(AutoTokenizer, "./Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9"),
            split='900,50,50',
            seed=2025,
    )
    recipe.trainer.plugins.grad_reduce_in_fp32 = False
    recipe.optim.config.use_precision_aware_optimizer = True
    recipe.model.config.cross_entropy_fusion_impl = "te"
    ddp_config=run.Config(
            DistributedDataParallelConfig,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
    )
    recipe.trainer.strategy.ddp=ddp_config
    recipe.trainer.strategy.fsdp = "megatron"
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    recipe.model.config.gradient_accumulation_fusion = False
    recipe.model.config.masked_softmax_fusion=True
    recipe.model.config.cross_entropy_loss_fusion = True
    recipe.model.config.apply_rope_fusion = True
    recipe.model.config.moe_permute_fusion = True
    recipe.model.config.bias_dropout_fusion = True
    recipe.model.config.bias_activation_fusion = True
    recipe.model.config.recompute_granularity = None
    recipe.model.config.recompute_method = None
    recipe.model.config.recompute_num_layers = None
    # 完全关闭保存ckpt，减少io开销
    recipe.log.ckpt = None
 
    recipe.trainer.callbacks = []
    recipe.trainer.callbacks=[run.Config(TimingCallback, log_tokens_per_sec = True)]
    recipe.trainer.callbacks.append(run.Config(MegatronTokenDropCallback))
    
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
    run.run(recipe, executor=executor, name="qwen3_30b_a3b_pretraining")

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    pl.seed_everything(2025, workers=True, verbose=True)
    set_seed(2025)
    run_pretraining()