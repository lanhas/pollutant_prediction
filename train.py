# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from network import PollutantPredModel_BP, PollutantPredModel_LSTM
from torch.utils.data import DataLoader
from dataloader import get_dataset
from torch import optim
from visdom import Visdom

# 超参数设置
arch_type = 'bp'
epochs = 200
lr = 1e-5
batch_size = 256
lstm_batch = 24


input_features = 27
output_features = 6
train_rate = 0.8
seq_len = 24 * 3

vis_port = 12370
vis_env = 'main'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if arch_type == 'bp':
    net = PollutantPredModel_BP(input_features, output_features).to(device)
elif arch_type == 'lstm':
    net = PollutantPredModel_LSTM(input_features, output_features, lstm_batch).to(device)
opt = optim.SGD(net.parameters(),lr=lr,momentum=0.9)

data_path = 'data/data.xlsx'
enable_vis = True
train_dst, val_dst = get_dataset(data_path, arch_type=arch_type, train_rate=train_rate, seq_len=seq_len)

if arch_type == 'bp':
    trainLoader = DataLoader(dataset=train_dst, batch_size=batch_size, shuffle=True, drop_last = True)
    valLoader = DataLoader(dataset=val_dst, batch_size=batch_size, shuffle=True, drop_last = True)
if arch_type == 'lstm':
    trainLoader = DataLoader(dataset=train_dst, batch_size=lstm_batch, shuffle=True, drop_last = True)
    valLoader = DataLoader(dataset=val_dst, batch_size=lstm_batch, shuffle=True, drop_last = True)

# 可视化设置
vis = Visdom(port=12370) if enable_vis else None
# 等间隔调整学习率
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = 30, gamma = 0.1, last_epoch=-1)
loss = nn.MSELoss()
# setup model path
model_folder = 'checkpoints'

# Restore
loss_his_train = []
loss_his_val = []
acc_his = []

def train():
    best_loss = 1
    name = ['tarin_loss', 'val_loss']
    for epoch in range(epochs):
        loss_train, loss_val = run_epoch(arch_type, epoch)
        loss_his_train.append(loss_train)
        loss_his_val.append(loss_val)

        if epoch % 10 == 0:
            print("Epoch:{}".format(epoch))
            print("train loss: {}".format(loss_train))
            print("val loss: {}".format(loss_val))
        if loss_val < best_loss and epoch > 50:
            best_loss = loss_val
            torch.save(net.state_dict(), 'checkpoints/Net_{}_Epopch{}-loss{}.pth'.format(arch_type,epoch, loss_val))
        vis.line(np.column_stack((loss_train, loss_val)), [epoch], win='train_log', update='append', opts=dict(title='losses', legend=name))

def run_epoch(arch_type, epoch):
    train_loss = 0
    val_loss = 0
    # train
    net.train()
    for _, (batch_x, batch_y) in enumerate(trainLoader):
        batch_x = torch.as_tensor(batch_x, dtype=torch.float32).to(device)
        batch_y = torch.as_tensor(batch_y, dtype=torch.float32).to(device)
        if arch_type == 'bp':
            out_puts = net(batch_x)
        elif arch_type == 'lstm':
            h0, c0 = net.h0().to(device), net.c0().to(device)
            _, h, c, out_puts = net(batch_x, h0, c0)
            batch_y = batch_y.reshape(-1, output_features)
        loss_step = loss(out_puts, batch_y)
        opt.zero_grad()
        loss_step.backward()
        opt.step()
        train_loss += float(loss_step.item())
    # val
    net.eval()
    with torch.no_grad():
        for _, (batch_x, batch_y) in enumerate(valLoader):
            batch_x = torch.as_tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.as_tensor(batch_y, dtype=torch.float32).to(device)
            if arch_type == 'bp':
                output = net(batch_x)
            elif arch_type == 'lstm':
                h0, c0 = net.h0().to(device), net.c0().to(device)
                _, h, c, output = net(batch_x, h0, c0)
                batch_y = batch_y.reshape(-1, output_features)
            loss_step = loss(output, batch_y)
            val_loss += float(loss_step.item())
        
    adjust_learning_rate(opt, epoch)
    loss_trained = train_loss / len(trainLoader)
    loss_val = val_loss / len(valLoader)
    return loss_trained, loss_val

def SGDlr(epoch, a1=lr, B=epochs):
    C = 200
    rest = epoch % B
    if rest < 0.5 * C:
        return a1
    elif rest < 0.9 * C:
        return (0.9 * C - rest) / (0.9 * C - 0.5 * C) * (a1 - 0.01 * a1)
    else:
        return 0.01 * a1

def adjust_learning_rate(optimizer, epoch):
    '''
    epoch:当前epoch
    fin_epoch：总共要训练的epoch数
    ini_lr:初始学习率
    lr:需要优化的学习率(optimizer中的学习率)
    '''
    lr = SGDlr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    train()