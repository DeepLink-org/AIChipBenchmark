

## install

reference: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/training_example_Hippocampus.md

```
git clone https://github.com/MIC-DKFZ/nnUNet.git

git checkout v1.6.5

#pip install -e . --user
export PTYHONPATH=`pwd`/nnunet:$PTYHONPATH

```

## dataset:
https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2


```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C' -O task04.tar

tar -xvf task04.tar

# get Task04_Hippocampus

export nnUNet_raw_data_base=`pwd`

export nnUNet_preprocessed=`pwd`/preprocessed

export RESULTS_FOLDER=`pwd`/result

mkdir preprocessed
mkdir result

#nnUNet_convert_decathlon_task -i Task04_Hippocampus
python -u  nnunet/experiment_planning/nnUNet_convert_decathlon_task.py -i Task04_Hippocampus

# The converted dataset can be found in $nnUNet_raw_data_base/nnUNet_raw_data


# nnUNet_plan_and_preprocess -t 4

python -u nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 4
#（上面的4就是数据集编号）


```

## train

```
# (不使用混合精度训练)
srun -p caif_dev --gres=gpu:1 python -u nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 4 0 --fp32

srun -p caif_dev --gres=gpu:8 --ntasks-per-node 8 --ntasks 16 --exclusive --job-name unet16gpu python -u nnunet/run/run_training_DDP.py 3d_fullres nnUNetTrainerV2_DDP 4 0 --fp32

```