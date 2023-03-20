import pandas as pd
import os
import sys
import subprocess

srun_cmd = "subprocess.check_output(['./cuda_ops/build/gemm', '{m}', '{k}', '{n}', '{trans1}', '{trans2}', '{datatype}'])"

if __name__ == '__main__':
    fname = sys.argv[1]
    datatype = int(sys.argv[2])
    validate = bool(int(sys.argv[3]))

    df = pd.read_csv(fname)

    if not validate:
        df['time'] = None
        df['score'] = None


    for i in df.index:
        m = df['M'][i]
        k = df['K'][i]
        n = df['N'][i]
        trans1 = df['transA'][i]
        trans2 = df['transB'][i]
        cmd_str = srun_cmd.format(m=m,
                                  k=k,
                                  n=n,
                                  trans1=trans1,
                                  trans2=trans2,
                                  datatype=datatype)
        out = eval(cmd_str)
        try:
            time = float(out.decode())
        except ValueError:
            print('Failed to decode the output.')
        time = format(time / 1000, '.3f')
        if not validate:
            df.at[i, 'baseline'] = time
        else:
            df.at[i, 'time'] = time
            df.at[i, 'score'] = round(
                float(df.at[i, 'baseline']) / float(time), 2)

    df.to_csv(fname, index=False)
