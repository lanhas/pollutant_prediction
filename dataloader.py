# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class PollutionDataset(data.Dataset):
    def __init__(self, data_input, data_output) -> None:
        self.input = data_input
        self.output = data_output

    def __getitem__(self, index):
        return self.input[index], self.output[index]

    def __len__(self):
        return self.input.shape[0]

def get_dataset(data_path, arch_type, train_rate=0.8, seq_len=72):
    df = pd.read_excel(data_path)
    scaler_input = MinMaxScaler()
    scaler_output = MinMaxScaler()
    df_input = df.iloc[:, :-6]
    df_output = df.iloc[:, -6:]
    norm_input = scaler_input.fit_transform(df_input)
    norm_output = scaler_output.fit_transform(df_output)
    joblib.dump(scaler_input, 'scaler_input.pkl')
    joblib.dump(scaler_output, 'scaler_output.pkl')

    if arch_type == 'bp':
        data_input = norm_input
        data_output = norm_output
    if arch_type == 'lstm':
        data_input = []
        data_output = []
        for i in range(norm_input.shape[0] - seq_len):
            temp_input = norm_input[i:i+seq_len]
            temp_output = norm_output[i:i+seq_len]
            data_input.append(temp_input)
            data_output.append(temp_output)
        data_input = np.array(data_input)
        data_output = np.array(data_output)

    input_train, input_test, output_train, output_test = train_test_split(data_input, data_output ,train_size=train_rate)
    train_dst, test_dst = PollutionDataset(input_train, output_train), PollutionDataset(input_test, output_test)
    return train_dst, test_dst
