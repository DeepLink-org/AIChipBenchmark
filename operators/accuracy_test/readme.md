# 算子精度测试

算子精度测试是进行不同软硬件环境下算子计算结果验证的工具。

本工具以Pytorch算子Python API为接口进行测试。

算子精度测试的流程分为：
1. 基准值生成
2. 在待测芯片上精度验证

基准值生成这一步骤已在英伟达A100 GPU上完成，其生成的数据在 `./a100_data` 目录。

算子精度验证参考 `cuda算子验证` 部分。

***Notice：*** 不同版本的Pytorch、不同的算子库实现、不同的硬件型号均会造成计算结果的不同，本仓库提供的代码和数据**仅供参考，不作为最终评测的标准**。


## 基准值生成
根据 `op_config.py` 的算子和参数，运行 `cuda_ground_truth_gen.py` ，生成算子的基准值数据。包括算子的输入、输出、参数和反向计算得到的梯度。

基准值生成会尝试运行`float32`和`float16`两种数据类型的计算，并保存在 `fp16`,`fp32`两个路径下。

运行命令：
```sh
python cuda_ground_truth_gen.py your/output/path
```

基准值数据将保存在 `your/output/path` 目录下。

### fp16 不支持的算子:
部分算子/数据类型是当前版本(pytorch1.8)CUDA不支持的，例如：

- ctc_loss
    - "ctc_loss_cuda" not implemented for 'Half'
- det
    - "lu_cuda" not implemented for 'Half'
- eig
    - "eig_cuda" not implemented for 'Half'
- svd
    - "svd_cuda_gesvdj" not implemented for 'Half'
- inverse
    - "inverse_cuda" not implemented for 'Half'

## cuda算子验证

根据`op_config.py` 的算子和参数，以及生成的基准测试数据，反向验证芯片上算子的精度，使用方法：

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python cuda_op_validate.py path/to/data path/to/output
```
精度验证的结果为 `your/output/path/cuda_val_result.csv`。

## logger环境变量
默认logging等级为`warning`, 通过环境变量 `PYLOGLEVEL` 可以指定logging等级; 例如 `export PYLOGLEVEL=INFO` 设置为logger等级为`info`。
