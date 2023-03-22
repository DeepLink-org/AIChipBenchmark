from wsgiref import validate
import pandas as pd
import os
import sys
import subprocess

srun_cmd = "subprocess.check_output(['./cuda_ops/build/conv', '{n}', '{c}', '{h}','{w}','{c_o}','{k_w}','{k_h}','{p_w}','{p_h}','{s_w}','{s_h}', '{datatype}'])"

if __name__=='__main__':
    fname=sys.argv[1]
    datatype=int(sys.argv[2])
    validate=bool(int(sys.argv[3]))

    df=pd.read_csv(fname)

    if not validate:
        df['time'] = None
        df['score'] = None

    for i in df.index:
        cmd_str=srun_cmd.format(
            n=df['N'][i],c=df['C'][i],h=df['H'][i],w=df['W'][i],
            c_o=df['OutC'][i], k_w=df['kw'][i], k_h=df['kh'][i],
            p_w=df['pw'][i],p_h=df['ph'][i],
            s_w=df['sh'][i], s_h=df['sv'][i],
            datatype=datatype

        )
        out=eval(cmd_str)
        out = out.decode('utf-8')
        outs = out.split('\n')
        forward_ms = float(outs[0])      # ms
        backward_weight_ms = float(outs[1]) # ms
        backward_data_ms = float(outs[2])    # ms
        total_time = forward_ms + backward_weight_ms + backward_data_ms
        if not validate:
            df.at[i, 'baseline'] = format(total_time, '.3f')
        else:
            df.at[i, 'time'] = time
            df.at[i, 'score'] = round(
                float(df.at[i, 'baseline']) / float(time), 2)

    avg_score = df['score'].mean()

    df.to_csv(fname, index=False)
