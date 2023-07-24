# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:15:58 2022
based historical time series and image sequences forecast
@author: 123
"""

import torch
import os
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

seq_size = 3


class IFE_LSTM(nn.Module):
    def __init__(self, input_nc=3, encode_dim=1024, lstm_hidden_size=1024,
                 seq_len=seq_size, num_lstm_layers=1, bidirectional=False):
        super(IFE_LSTM, self).__init__()
        self.seq_len = seq_len
        self.num_directions = 2 if bidirectional else 1
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.cnn = nn.Sequential(
            nn.Conv2d(input_nc, 12, 3, 1, 1),  # 32*64*64
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            # 32*63*63
            nn.Conv2d(12, 24, 3, 1, 1),  # 64*32*32
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),
            # 64*31*31
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2))
        self.fc1 = nn.Linear(24576, encode_dim)
        self.fc2 = nn.Linear(1, encode_dim)
        self.fc = nn.Linear(2048, encode_dim)
        self.lstm = nn.LSTM(encode_dim, lstm_hidden_size, num_lstm_layers, batch_first=True)
        self.linear = nn.Linear(self.lstm_hidden_size, 1)

    def init_hidden(self, x1):
        batch_size = x1.size(0)
        h = x1.data.new(
            self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        c = x1.data.new(
            self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        return Variable(h), Variable(c)
            # [batchsize*seqsize ,25]

    def forward(self, x1, x2):
        # x1.shape [batchsize,seqsize,3,128,128]
        b = x1.size(0)
        x1 = x1.view(b * seq_size, 3, 128, 128)  # x.shape[batchsize*seqsize,3,128,128]
        # [batchsize*seqsize, 3, 128, 128] -> [batchsize*seqsize, 1024,1,1]
        x1 = self.cnn(x1)
        # [batchsize * seqsize, 1024, 1, 1]-> [batchsize*seqsize, 1024]
        x1 = x1.view(b * seq_size, -1)
        # [batchsize * seqsize, 1024]
        # x2.shape [batchsize,seqsize]
        x2 = x2.view(b * seq_size, -1)
        x2 = self.fc2(x2)
        x1 = self.fc1(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc(x)
        # [batchsize*seqsize ,1024]
        x = x.view(-1, seq_size, x.size(1))
        h0, c0 = self.init_hidden(x)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = output.contiguous().view(b * seq_size, -1)
        pred = self.linear(output)  # pred(150, 1)
        pred = pred.view(b, seq_size, -1)  # (5, 24, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred


if __name__ == '__main__':
    model = IFE_LSTM()
    x1 = torch.randn(10, 5, 3, 128, 128)
    x2 = torch.randn(10, 5)
    y = model(x1, x2)
    print(y.shape)
