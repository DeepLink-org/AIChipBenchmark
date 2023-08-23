#/bin/bash
set -e

if [ $# != 1 ] ; then
    echo "USAGE: bash test_ckp.sh MODEL_NAME"
    echo " e.g.: bash test_ckp.sh 7B"
    exit 1;
fi

#7B
cmd_7b="srun -p pat_rd --gres=gpu:1 python profile_ckp_time.py  /mnt/lustre/share_data/PAT/datasets/llama/llama_v1_workspace"

#65B
cmd_65b="srun -p pat_rd --gres=gpu:8 python profile_ckp_time.py  /mnt/lustre/share_data/PAT/datasets/llama/65B/llama_v1_workspace/ --tp 8"

cmd=""
if [[ "$1" == "7B" ]]; then
    cmd=$cmd_7b
elif [[ $1 == '65B' ]]; then
    cmd=$cmd_65b
else 
    echo "USAGE: bash test_ckp.sh 7B/65B"
    exit 1;
fi
echo "cmd is: $cmd"

echo "warm up"
for((i=1;i<=5;i++));  
do   
    result=`$cmd`
    echo $result
done 
echo "warm up end"


arr=()
for((i=1;i<=10;i++));  
do   
    result=`$cmd`
    res=$(echo "scale=2; $result" | bc)
    arr+=($res)
done 

sum=0
for (( i=0;i<${#arr[*]};i++))
do
        echo ${arr[$i]}
        sum=$(echo "$sum + ${arr[$i]}" | bc)
done
total=${#arr[*]}

echo "ckp loading time avg is: "
echo "scale=2; $sum/$total" | bc
