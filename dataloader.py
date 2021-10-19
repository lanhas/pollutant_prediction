# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class PollutionDataset(data.Dataset):
    def __init__(self, root, arch_type, train=True, test_size=0.2, time_step=72) -> None:
        self.arch_type = arch_type
        self.time_step = time_step
        data = np.array(pd.read_excel(root))
        # 归一化
        minmax_scaler_input = MinMaxScaler()
        minmax_scaler_output = MinMaxScaler()
        data_input = data[:, :-6]
        data_output = data[:, -6:]

        data_input = minmax_scaler_input.fit_transform(data_input)
        data_output = minmax_scaler_output.fit_transform(data_output)
        joblib.dump(minmax_scaler_input, 'scaler_input.pkl')
        joblib.dump(minmax_scaler_output, 'scaler_output.pkl')
        data = np.concatenate((data_input, data_output), axis=1)
        data_train, data_test = split_data(data, test_size)
        if train == True:
            self.data = data_train
        else:
            self.data = data_test

    def __getitem__(self, index):
        if self.arch_type == 'bp':
            return self.data[index,:-6], self.data[index, -6:]
        elif self.arch_type == 'lstm':
            return self.data[index*self.time_step: (index+1)*self.time_step, :-6], self.data[index*self.time_step: (index+1)*self.time_step, -6:]

    def __len__(self):
        if self.arch_type == 'bp':
            return len(self.data)
        elif self.arch_type == 'lstm':
            return len(self.data)//self.time_step

def split_data(data, test_size):
    data_num = len(data) // 3
    data_a = data[:data_num, :]
    data_b = data[data_num:data_num*2]
    data_c = data[data_num*2:]
    da_train, da_test = train_test_split(data_a, test_size=test_size, shuffle=False)
    db_train, db_test = train_test_split(data_b, test_size=test_size, shuffle=False)
    dc_train, dc_test = train_test_split(data_c, test_size=test_size, shuffle=False)
    data_train = np.concatenate((da_train, db_train, dc_train), axis=0)
    data_test = np.concatenate((da_test, db_test, dc_test), axis=0)
    return data_train, data_test
