#/bin/bash
set -e

PARALLEL=2

tempfifo=$$.fifo
trap "exec 1000>&-;exec 1000<&-;exit 0" 2
mkfifo $tempfifo
exec 1000<>$tempfifo
rm -rf $tempfifo
for ((i=1; i<=$PARALLEL; i++))
do
    echo "testing"
    echo >&1000
done

# make logging dir
time=$(date "+%Y%m%d-%H%M%S")
echo "${time}"
mkdir ${time}

for line in `(tail -n +2 config.csv)`
do

    array=(${line//,/ })
    model=${array[0]}
    concurrency=${array[1]}
    Batchsize=${array[2]}
    Input=${array[3]}
    Output=${array[4]}

    read -u1000
    {
        echo model: $model
        echo concurrency: $concurrency
        echo Batchsize: $Batchsize
        echo Input: $Input
        echo Output：$Output
        echo ""

        if [ "$model" == "7B" ]; then
            cmd_7b="
                srun -p pat_rd --gres=gpu:1 python profile_generation.py \
                /mnt/lustre/share_data/PAT/datasets/llama/llama_v1_workspace \
                --concurrency $concurrency \
                --input_seqlen $Input \
                --output_seqlen $Output \
                >> ${time}/result_log_${model}_${concurrency}_${Input}_${Output}
                "
            echo $cmd_7b
            eval $cmd_7b
        elif [ "$model" == "65B" ]; then
            cmd_65b="
                srun -p pat_rd --gres=gpu:8 python profile_generation.py \
                /mnt/lustre/share_data/PAT/datasets/llama/65B/llama_v1_workspace \
                --concurrency $concurrency \
                --input_seqlen $Input \
                --output_seqlen $Output \
                --test_round 10 \
                --tp 8 \
                >> ${time}/result_log_${model}_${concurrency}_${Input}_${Output}
                "
            echo $cmd_65b
            eval $cmd_65b
        else
            echo "Invalid model config"
            exit 1;
        fi
        echo >&1000
    } &
done
wait
echo "Testing Finished!!!!!!!!!!"
