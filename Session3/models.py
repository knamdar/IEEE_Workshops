# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 23:23:45 2019

@author: Ernest (Khashayar) Namdar
"""
import torch.nn as nn


class CNN(nn.Module):
    """CREATE MODEL CLASS"""
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1,
                              out_channels=16,
                              kernel_size=3,
                              stride=1,
                              padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=3,
                              stride=1,
                              padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        # FC
        out = self.fc1(out)

        return out
