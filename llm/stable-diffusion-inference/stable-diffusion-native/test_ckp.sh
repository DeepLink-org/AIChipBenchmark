#/bin/bash
set -e

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cmd="srun -p pat_rd --gres=gpu:1  python scripts/txt2img_ckp.py"

echo "cmd is: $cmd"

echo "warm up"
for((i=1;i<=3;i++));  
do   
    eval $cmd
done 
echo "warm up end"

echo "testing start"
arr=()
for((i=1;i<=5;i++));  
do   
    eval $cmd
done 
echo "testing finished"
