# -*- coding: utf-8 -*-
import numpy as np
import torch
import joblib
import pandas as pd
from torch._C import device
from network import PollutantPredModel_BP, PollutantPredModel_LSTM

# 超参数设置
model_path_lstm = 'checkpoints/Net_lstm_Epopch533-loss0.006075630596439753.pth'
model_path_bp = 'checkpoints/Net_bp_Epopch187-loss0.011359991457570216.pth'
input_features = 27
output_features = 6
lstm_batch = 1

def data_resove(data):
    datas_location = []
    datas_dayMean = []
    for i in range(3):
        da = np.round(data[i*72:(i+1)*72])
        datas_location.append(da)
    for idx, data in enumerate(datas_location):
        daysMean = []
        for i in range(3):
            data_day = data[i*24:(i+1)*24]
            data_mean =np.round(np.mean(data_day, axis=0), 2)
            daysMean.append(data_mean)
        datas_dayMean.append(daysMean)
    return datas_location, datas_dayMean

def result_print(datas_location, datas_dayMean, save=False):
    location_name = ["A", "B", "C"]
    day_name = [1, 2, 3]
    for idx,data_location in enumerate(datas_location):
        print("监测点{}结果：\n{}".format(location_name[idx],data_location))
        if save:
            np.savetxt("result{}.csv".format(location_name[idx]), data_location, delimiter=',')
    for idx, data_daysMean in enumerate(datas_dayMean):
        print("监测点{}:".format(location_name[idx]))
        for day_index, dayMean in enumerate(data_daysMean):
            print("第{}日平均：\n{}".format(day_name[day_index], dayMean))

def predict(data, arch_type):
    if arch_type == 'bp':
        model = PollutantPredModel_BP(input_features, output_features)
        checkpoint = torch.load(model_path_bp, map_location=torch.device('cpu'))
    elif arch_type == 'lstm':
        model = PollutantPredModel_LSTM(input_features, output_features, lstm_batch)
        checkpoint = torch.load(model_path_lstm, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint)
    minmax_scaler_input = joblib.load('scaler_input.pkl')
    minmax_scaler_output = joblib.load('scaler_output.pkl')
    data_input = minmax_scaler_input.transform(data)
    data_tensor = torch.as_tensor(data_input, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        model = model.eval()
        if arch_type == 'bp':
            output = model(data_tensor)
        elif arch_type == 'lstm': 
            h0, c0 = model.h0().to(device), model.c0().to(device)     
            _, h, c, output = model(data_tensor, h0, c0)
    predicts = minmax_scaler_output.inverse_transform(output)
    # predicts = np.absolute(predicts)

    datas_location, datas_dayMean = data_resove(predicts)
    result_print(datas_location, datas_dayMean)

if __name__ == "__main__":
    data = pd.read_excel('data/predict_data.xlsx')
    data = np.array(data)
    predict(data, arch_type='bp')