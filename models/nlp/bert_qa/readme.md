```sh
source /mnt/cache/share/platform/env/pt1.7v1


git clone https://github.com/huggingface/transformers.git

git checkout 2d91e3c30410e0e17e3247323e46c443586eb508

cd transformers

pip install -e .

# not sure if this is needed
pip install tokenizers datasets --user

cd examples/pytorch/question-answering


# initial test
srun -p caif_dev --exclusive --gres=gpu:8 python run_qa.py --model_name_or_path bert-base-uncased --dataset_name squad --do_eval --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp

```

and get:

```
{'task': {'name': 'Question Answering', 'type': 'question-answering'}, 'dataset': {'name': 'squad', 'type': 'squad', 'args': 'plain_text'}}
***** eval metrics *****
  eval_exact_match = 0.2838
  eval_f1          = 7.3981
  eval_samples     =  10784

```

do train:
```
srun -p caif_dev --exclusive --gres=gpu:1 python run_qa.py --model_name_or_path bert-base-uncased --dataset_name squad --do_train --do_eval --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir /mnt/lustre/shidongxing/work/tmp
```

get:

```
{'train_runtime': 2264.14, 'train_samples_per_second': 78.197, 'train_steps_per_second': 6.516, 'train_loss': 0.981123586196765, 'epoch': 2.0}
***** train metrics *****
  epoch                    =        2.0
  train_loss               =     0.9811
  train_runtime            = 0:37:44.13
  train_samples            =      88524
  train_samples_per_second =     78.197
  train_steps_per_second   =      6.516

***** eval metrics *****
  epoch            =     2.0
  eval_exact_match = 80.8231
  eval_f1          = 88.2829
  eval_samples     =   10784
```

# distributed training:

```
srun -p caif_dev --exclusive --gres=gpu:4 python -m torch.distributed.launch --nproc_per_node 4 run_qa.py --model_name_or_path bert-base-uncased --dataset_name squad --do_train --do_eval --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir /mnt/lustre/shidongxing/work/tmptrain_n4
```