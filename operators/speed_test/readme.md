
# 算子性能测试
算子性能测试包括CUDA中GEMM、Conv2d算子性能测试和长尾算子性能测试。
## GEMM、Conv2d算子性能测试
包括GEMM算子、Conv2d算子在FP16(使用tensor core)、FP32条件下进行性能测试。

流程说明如下：

  * **编译**: 进入cuda_ops目录，执行相关编译；
  * **生成基准值**：生成基准值，`单独运行`指测试单个测试数据，可得到相应结果；`批量运行`指读取`*.csv`文件里的测试数据，批量运行并将结果写入`*.csv`文件`baseline`列中。**注意**：如果已经有基准值，则跳过该步骤。
  * **测试**：测试芯片GEMM和Conv2d算子，根据`*csv`的测试数据和`generate gemm and conv baseline`步骤中的结果`baseline`，得到测试结果和相较于`baseline`的得分（比值），结果分别写入`.csv`文件`time`和`score`列中。

### 1 编译

requires cudnn8
```
cd cuda_ops

srun -p $partition -n 1 -N 1 mkdir -p build && cd build && cmake .. && make

```
### 2 生成基准值
#### 单独运行
```
cd cuda_ops

srun -p $partition --gres=gpu:1 --exclusive build/gemm m k n trans1 trans2 datatype

srun -p $partition --gres=gpu:1 --exclusive build/conv n c h w c_out k_w k_h pad_w pad_h stride_w stride_h datatype

```

参数解释：
`datatype`: `16`表示fp16,`32`表示fp32

`trans1`、`trans2`: `0`表示不转置，`1`表示转置
#### 批量运行

以下分別生成卷积算子和GEMM算子在数据类型为FP16和FP32的基准值。
```
srun -p $partition --gres=gpu:1 --exclusive python test_conv.py conv_f16.csv 16 0

srun -p $partition --gres=gpu:1 --exclusive python test_conv.py conv_f32.csv 32 0

srun -p $partition --gres=gpu:1 --exclusive python test_gemm.py gemm_f16.csv 16 0

srun -p $partition --gres=gpu:1 --exclusive python test_gemm.py gemm_f32.csv 32 0
```
从`*.csv`文件读取参数并写入生成的基准值，`16/32`表示数据类型，`0`表示生成基准值。

运行的结果在 `.csv`中，例如 `gemm_f16.csv` 是在A100 GPU 在float16下运行的结果。

### 3 测试
以下分别生成Conv2d算子和GEMM算子在数据类型为FP16和FP32的测试值，并得到相对基准值的得分。
```
srun -p $partition --gres=gpu:1 --exclusive python test_conv.py conv_f16.csv 16 1

srun -p $partition --gres=gpu:1 --exclusive python test_conv.py conv_f32.csv 32 1

srun -p $partition --gres=gpu:1 --exclusive python test_gemm.py gemm_f16.csv 16 1

srun -p $partition --gres=gpu:1 --exclusive python test_gemm.py gemm_f32.csv 32 1
```
从`*.csv`文件读取参数和基准值并写入测试组和得分，`16/32`表示数据类型，`1`表示测试。

运行的结果在 `.csv`中。

## 长尾算子性能测试

长尾算子测试流程包括设置环境变量、生成基准值(如果已有基准值数据，则跳过该步）以及验证测试。

`LongTail-Bench`是长尾算子的Pytorch实现代码，仅供参考。

测试流程如下：

### 1 设置环境变量
```
cd LongTail-Bench

export PYTHONPATH=./long_tail_bench:$PYTHONPATH
```
### 2 生成基准值

在gpu上生成基准:
```
srun -p $partition --gres=gpu:1 --exclusive python ./long_tail_bench/api/api.py -f ../longtail_perf.csv --outcsv path/to/ltout_gpu.csv
```

在cpu上生成基准:
* 执行转化脚本
```bash
sh script_for_cpu.sh
```
转化脚本会生成samples-bak,存放原来的samples，新的samples文件夹已经适配了cpu,不再支持gpu测试。如果要再次测试gpu，可以恢复samples-bak为samples，在执行在gpu上测试的脚本。

* 生成基准
```bash
 DEVICE_CPU=1 srun -p $partition --exclusive python ./long_tail_bench/api/api.py -f ../longtail_perf.csv --outcsv path/to/ltout_cpu.csv
```
### 3 测试
```
srun -p $partition --gres=gpu:1 --exclusive python ./long_tail_bench/api/api.py -f ../longtail_perf.csv --outcsv path/to/ltout.csv --validate
```
计算的结果存在输出参数`path/to/ltout.csv`中

