
partition=$1
bin=$2  # path to osu_nccl_allreduce
output=$3

mkdir -p logs

function run_one()
{
    ngpu=$1
    num=$2  # num of floats
    if [ $ngpu -lt 8 ]
    then
        nnode=1
        nt_node=$ngpu
    else
        nnode=`expr $ngpu / 8`
        nt_node=8
    fi

    num=`expr $num \* 4`
    echo "num: $num"
    sh comm_sbatch.sh $ngpu $bin $num > logs/comm_${ngpu}_${num}.out
    #sbatch -o logs/comm_${ngpu}_${num}.out -p $partition --exclusive --mem=0 -n $ngpu -N $nnode --gres=gpu:$nt_node --ntasks-per-node $nt_node --cpus-per-task 10 --job-name=comm_test comm_sbatch.sh $ngpu $bin $num
    sleep 20s
}



# block until allfinish
function wait_task()
{
    # query squeue, make sure 'openmm_models' are finished
    ret=`squeue --user $USER -p $partition  | grep comm_test`
    # echo $ret

    while [ "$ret" != "" ]
    do
        echo "waiting for openmm test to finish"
        sleep 1m
        ret=`squeue --user $USER -p $partition  | grep comm_test`
    done
}

function test_iter()
{
    for ngpu in 2 4 8
    do
        for num in 300000 3097600 6553600 16777216 134217700
        do
            run_one $ngpu $num
        done
    done
    wait_task
    wait_task
    python parse_comm_result.py logs $output
    #srun -p $partition -n 1 --cpus-per-task 10 python parse_comm_result.py logs $output
}

test_iter
# test_iter

