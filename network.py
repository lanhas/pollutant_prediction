# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import LSTM

def block(input_channel, output_channel, Nprmalize=True, Drop=False):
    layers = []
    layers.append(nn.Linear(input_channel, output_channel, bias=False))
    if Nprmalize==True:
        layers.append(nn.BatchNorm1d(output_channel))
    if Drop:
        layers.append(nn.Dropout(0.2))
    layers.append(nn.ReLU())
    return layers

class PollutantPredModel_LSTM(nn.Module):
    def __init__(self, input_channel, output_channel, batch):
        super().__init__()
        num_input = input_channel
        self.num_hidden = 27
        self.num_output = output_channel
        self.batch = batch
        self.rnn = LSTM(
            input_size=num_input,
            hidden_size=self.num_hidden,
            num_layers=1,
            batch_first=True)
        self.out = nn.Sequential(
            *block(self.num_hidden, 128),
            *block(128, 64),
            nn.Linear(64, output_channel),
        )
        self.drop = nn.Dropout(p=0.2)
    
    def forward(self, x, h, c):
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)

        # output shape: (seq_len, batch, num_directions * hidden_size)
        lstm_out, (hn, cn) = self.rnn(x, (h, c))
        output = self.out(lstm_out.contiguous().view(-1, self.num_hidden))
        return lstm_out, hn,  cn, output

    def h0(self):
        return torch.randn((1, self.batch, self.num_hidden))

    def c0(self):
        return torch.randn((1, self.batch, self.num_hidden))

class PollutantPredModel_BP(nn.Module):
    def __init__(self, input_channel=22, output_channel=6):
        super(PollutantPredModel_BP, self).__init__()
        self.model = nn.Sequential(
            *block(input_channel, 128, Drop=True),
            *block(128, 256, Drop=True),
            *block(256, 128, Drop=True),
            nn.Linear(128, output_channel),
        )

    def forward(self, x):
        output = self.model(x)
        return output

if __name__ == "__main__":
    net = PollutantPredModel_BP(22, 6)
    print(net)