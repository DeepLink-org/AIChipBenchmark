#/bin/bash
set -e

PARALLEL=8

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
    Batchsize=${array[0]}
    H=${array[1]}
    W=${array[2]}
    # Input=${array[3]}
    # Output=${array[4]}

    read -u1000
    {
        echo Batchsize: $Batchsize
        echo H: $H
        echo W: $W
        # echo Input: $Input
        # echo Output：$Output
        echo ""

        export HF_DATASETS_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        cmd_7b="
        srun -p pat_rd --gres=gpu:1 \
        python scripts/txt2img.py --prompt \
        \"Emma Watson as a powerful mysterious sorceress, casting lightning magic, detailed clothing, digital painting\" \
        --plms --n_samples $Batchsize --skip_grid  --skip_save --H $H --W $W
        >> ${time}/result_log_${Batchsize}_${H}_${W}
        "
        echo $cmd_7b
        eval $cmd_7b
        echo >&1000
    } &
done
wait
echo "Testing Finished!!!!!!!!!!"
