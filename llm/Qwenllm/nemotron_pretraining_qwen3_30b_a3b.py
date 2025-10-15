import nemo_run as run

from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.utils.exp_manager import TimingCallback

def configure_recipe(nodes: int = 1, gpus_per_node: int = 8):

    recipe = llm.qwen3_30b_a3b.pretrain_recipe(num_nodes=nodes, num_gpus_per_node=gpus_per_node)
    recipe.data.tokenizer = run.Config(AutoTokenizer, "./models--Qwen--Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9")
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
    run.run(recipe, executor=executor, name="qwen3_30b_a3b_pretraining")

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()