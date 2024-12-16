#!/bin/bash


filename="path/to/your/full-finetune/file.log"
start_iter=5
end_iter=22

python calc.py "$filename" "$start_iter" "$end_iter"

filename="path/to/your/qlora-finetune/file.log"
start_iter=5
end_iter=382
python calc.py "$filename" "$start_iter" "$end_iter"