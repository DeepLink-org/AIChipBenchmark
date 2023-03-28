import pandas as pd
from torch import nn
def get_conv_config(path=f'../speed_test/conv_f16.csv'):
    df = pd.read_csv(path)
    col = ['NO', 'W', 'H', 'C', 'N', 'OutC', 'kw', 'kh', 'pw', 'ph', 'sh', 'sv']
    df = df.loc[:, col]
    df = df.astype('int32')
    conv_config = {}
    row_num = df.shape[0]

    for row in range(row_num):
        data = df.iloc[row,:]
        key = 'conv2d_' + str(data['NO'])
        conv_config[key] = (
            nn.Conv2d(
                in_channels=data['C'], 
                out_channels=data['OutC'], 
                kernel_size=(data['kw'], data['kh']), 
                stride=(data['sh'], data['sv']), 
                padding=(data['pw'], data['ph']), 
            ),
            ((int(data['N']), int(data['C']), int(data['H']), int(data['W'])))
        )
    return conv_config
    