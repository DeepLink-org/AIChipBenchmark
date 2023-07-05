#!/bin/bash

#SBATCH --job-name=openmm_models        # name
#SBATCH --time 720:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --cpus-per-task 5


CONFIG=$1
WORK_DIR=$2
EXTRA_ARGS=${@:3}

# EXTRA_ARGS:
# "--cfg-options runner.max_epochs=1"

echo "python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR}  $EXTRA_ARGS"

srun --kill-on-bad-exit=1 \
    --exclusive \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" $EXTRA_ARGS
