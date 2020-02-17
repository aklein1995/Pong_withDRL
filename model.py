#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:05:56 2020

@author: alain
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride  (round up if not an integer)
        # output --> [(W−K+2P)/S]+1
        """
        self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=2) # output: 4 matrix of 40x40
        self.conv2 = nn.Conv2d(4, 1, kernel_size=2, stride=2) # output: 1 matrix of 20x20
        self.pool = nn.MaxPool2d(2,2) # output: 1 matrix of 10x10
        self.size = 1*10*10
        """
        # CONV LAYERS
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2) # output: 4 matrix of 38x38
        self.conv2 = nn.Conv2d(4, 8, kernel_size=4, stride=2) # output: 8 matrix of 18x18
        self.conv3 = nn.Conv2d(8, 8, kernel_size=4, stride=2) # output: 8 matrix of 9x9
        self.size = 8*8*8
        # DENSE-LINEAR LAYERS
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,self.size) # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.sig(self.fc2(x))
        return x