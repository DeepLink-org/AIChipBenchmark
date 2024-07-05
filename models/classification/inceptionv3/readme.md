# model
gitlab.bj.sensetime.com/platform/ParrotsDL/parrots.example

e8fe98cd

参考配置：inception.yaml

# env

```
export PYTHONPATH=/mnt/lustre/share/pymc/new:$PYTHONPATH


#dataset.py

    def _init_tcs_client(self):
        if not self.initialized:
            client_config_file = "/mnt/lustre/share/pymc/mc.conf"
            self.tcsclient = mc.TcsClient.GetInstance(
            '', client_config_file)
            self.initialized = True

    def __getitem__(self, index):
        filename = self.root + '/' + self.metas[index][0]
        cls = self.metas[index][1]

        # tcs_server
        self._init_tcs_client()
        value = mc.pyvector()
        self.tcsclient.Get(filename, value)
        value_buf = mc.ConvertBuffer(value)
        buff = io.BytesIO(value_buf)
        with Image.open(buff) as img:
            img = img.convert('RGB')

```

# run


```
# train.sh modified:
srun -p $partition --job-name=$jobname \
    --gres=gpu:$g -n$gpus --ntasks-per-node=$g --exclusive \
    python -u main.py --config $cfg \
    2>&1 | tee log/train_$jobname.log-$now

# 16GPUs, lr=0.2

sh train.sh configs/inception.yaml incepaccu caif_dev 16
```
