import os

from utils.utils import read_base

with read_base():
    from configs._base_.default_runtime import *  # pylint: disable=wildcard-import,unused-wildcard-import
    from configs._base_.models.llama.llama_7B import *  # pylint: disable=wildcard-import,unused-wildcard-import
    from configs._base_.monitors.base import *  # pylint: disable=wildcard-import,unused-wildcard-import

if "JOB_NAME" in os.environ:
    JOB_NAME = os.environ["JOB_NAME"]
else:
    JOB_NAME = os.path.basename(__file__).split(".py")[0]

# If set, will enable debug mode
# In non-debug mode, all the local changes are requested to be committed, otherwise the training will not start
DEBUG = 1

# If set, will enable sft training
DO_SFT = False

ENABLE_SAVE_CKPT = False

# Two settings: "streaming" and "preprocessed"
# If set to "streaming", will use streaming dataset (on-the-fly tokenize)
# If set to "preprocessed", will use pre-tokenized dataset
DATASET_TYPE = "preprocessed"

# Dataset path
TRAIN_FOLDER: str = "/mnt/petrelfs/share_data/huangye.p/train_internlm/dataset_folder"
VALID_FOLDER: str = None

TOTAL_STEP = 200000
VALID_EVERY = 2000

MICRO_NUM = 4
VALID_MICRO_NUM = 1
GRADIENT_ACCUMULATION = MICRO_NUM

MICRO_BATCH_SIZE = 2  # packed_length = micro_batch_size * seq_len
SEQ_LEN = 4096
MIN_LENGTH = 50

# Truncation rules for the pack process.
# It is recommended to set it to `none` for pre-training and
# `complete` for fine-tuning tasks to keep the context intact.
PACK_DATASET_BREAK_MODE = "none"

# If set to -1, will use SEQ_LEN as the max length of each sample
MAX_LENGTH_PER_SAMPLE = -1  # Or set as SEQ_LEN

LEARNING_RATE = 4e-5
MIN_LEARNING_RATE = 4e-6
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.025
OPTIMIZER_WARMUP_STEP = 0

CHECKPOINT_EVERY = 1000
SNAPSHOT_FREQ = CHECKPOINT_EVERY // 4

OSS_NAME = "checkpoints_ssd_02"
OSS_IP = "10.135.7.249"  # P cluster
# OSS_NAME = "model_weights"
# OSS_IP = "10.140.14.252"  # S cluster

# Ckpt folder format:
#  fs: 'local: /mnt/nfs/XXX'
# oss: 'boto3: s3://model_weights/XXX'
SAVE_CKPT_FOLDER = f"boto3:s3://{OSS_NAME}.{OSS_IP}/{JOB_NAME}/"

# If you want to train from scratch, set LOAD_CKPT_FOLDER to None.
LOAD_CKPT_FOLDER = None
# LOAD_CKPT_FOLDER = "boto3:s3://checkpoints_ssd_02.10.135.7.249/llm_llama2/llama2_raw/llama-2-70b"
# NOTE: ckpt_type should be in "internlm", "llama", "fuxi", "newton", "maibao", "plato"
LOAD_CKPT_FOLDER_INFO = dict(path=LOAD_CKPT_FOLDER, content=["model"], ckpt_type="llama")

DATASET_WEIGHTS = {"en": 2.0}

ckpt = dict(
    # Save ckpt settings
    enable_save_ckpt=ENABLE_SAVE_CKPT,  # If set to True, will save ckpt to save_ckpt_folder.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save ckpt.
    # Save ckpt frequency
    checkpoint_every=CHECKPOINT_EVERY,
    oss_snapshot_freq=SNAPSHOT_FREQ,
    # Load ckpt settings
    auto_resume=False,  # If set to True, will auto-load the latest checkpoint in save_ckpt_folder.
    load_ckpt_info=LOAD_CKPT_FOLDER_INFO,  # If auto_resume is False, will load checkpoint from load_ckpt_folder.
    # Other infos
    async_upload=True,
    async_upload_tmp_folder=f"/dev/shm/internlm_tmp_ckpt_{JOB_NAME}/",
    stop_file_path=f"llm_alter/{JOB_NAME}.log",
)

data = dict(
    type=DATASET_TYPE,
    sft=DO_SFT,  # if do sft training, set True
    seq_len=SEQ_LEN,
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=MICRO_NUM,
    micro_bsz=MICRO_BATCH_SIZE,
    # defaults to the value of micro_num
    valid_micro_num=VALID_MICRO_NUM,
    # defaults to 0, means disable evaluate
    valid_every=VALID_EVERY,
    pack_sample_into_one=False,
    total_steps=TOTAL_STEP,
    skip_batches="",
    rampup_batch_size="",
    # Datasets with less than `MIN_LENGTH` will be discarded
    min_length=MIN_LENGTH,
    train_folder=TRAIN_FOLDER,
    valid_folder=VALID_FOLDER,
    vocab_file=VOCAB_FILE,  # pylint: disable=undefined-variable
    text_field="content",
    num_worker=4,
    gradient_accumulation=GRADIENT_ACCUMULATION,
    dataset_weights=DATASET_WEIGHTS,  # sample_data_weights
    break_mode=PACK_DATASET_BREAK_MODE,
    max_length_per_sample=MAX_LENGTH_PER_SAMPLE,
)

grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**14,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)

loss = dict(label_smoothing=0.0)

adam = dict(
    lr=LEARNING_RATE,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=WEIGHT_DECAY,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=OPTIMIZER_WARMUP_STEP,  # optimizer_warmup_step
    warmup_ratio=WARMUP_RATIO,
    eta_min=MIN_LEARNING_RATE,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    zero_overlap_communication=True,
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=True,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

# zero1 parallel:
#     1. if zero1 <= 0, The size of the zero process group is equal to the size of the dp process group,
#         so parameters will be divided within the range of dp.
#     2. if zero1 == 1, zero is not used, and all dp groups retain the full amount of model parameters.
#     3. zero1 > 1 and zero1 <= dp world size, the world size of zero is a subset of dp world size.
#         For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
# pipeline parallel (dict):
#     1. size: int, the size of pipeline parallel.
#     2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler.
# tensor parallel: tensor parallel size, usually the number of GPUs per node.
parallel = dict(zero1=dict(size=-1, fsdp=False), pipeline=dict(size=1, interleaved_overlap=True), tensor=2, sequence_parallel=False)

