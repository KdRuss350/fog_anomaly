import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


class Model(nn.Module):  #x_enc, x_mark_enc, x_dec, x_mark_dec,
    def __init__(self,configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        block1 = []
        block1 += [
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)  ###maxpooling每一组找最大值
        ]
        self.convblock1 = nn.Sequential(*block1)

        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        self.flatten = nn.Flatten()

        flatten_channel = 32 * self.seq_len // 4

        self.denseblock = nn.Sequential(
            nn.Linear(flatten_channel, 40),  # 根据计算得到的flat_dim
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )
        self.out = nn.Linear(40, self.pred_len.shape[1])


    def forward(self, x):
        ### 进来就是(32,96,1)
        # x = x.view(-1, seq_len, 1)###batch-size,features,steps,所以这里要view一下，要加上feature这个特征！
        x = x.permute(0,2,1)
        # print('view后的shape：',x.shape),变成(32,1,96)进入卷积
        x = self.convblock1(x)
        # print('第一层后shape:',x.shape)
        x = self.convblock2(x)
        #print('第二层后shape:', x.shape)
        x = self.flatten(x)
        #print('flatten后shape:', x.shape)
        x = self.denseblock(x)
        #print('denseblock后shape:', x.shape)
        x = self.out(x)
        #print('out后shape:', x.shape)
        x = x.unsqueeze(-1)

        return x  # 网络前向过程编写

