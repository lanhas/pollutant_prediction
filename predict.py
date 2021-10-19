# -*- coding: utf-8 -*-
import numpy as np
import torch
import joblib
import pandas as pd
from network import PollutantPredModel_BP, PollutantPredModel_LSTM

# 超参数设置
model_path_lstm = 'checkpoints/Net_lstm_Epopch459-loss0.00848061079159379.pth'
model_path_bp = 'checkpoints/Net_bp_Epopch108-loss0.011935063637793064.pth'
input_features = 27
output_features = 6
lstm_batch = 1

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
            h0, c0 = model.h0(), model.c0()
            _, h, c, output = model(data_tensor, h0, c0)
    predicts = minmax_scaler_output.inverse_transform(output)
    predicts = np.absolute(predicts)
    np.savetxt("resultC.csv", predicts, delimiter=',', fmt='%.04f')
    print(predicts)

if __name__ == "__main__":
    data = pd.read_excel('predict_data.xlsx')
    data = np.array(data)
    predict(data, arch_type='lstm')
        








