import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")  #ignore warnings

class cnn(nn.Module): # construction of netral network
    def __init__(self,num_classes):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d( # 1 224 224
                in_channels=3, # input rgb size
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2 # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Conv2d(  # 1 224 224
                in_channels=16,  # input rgb size
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2  # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16,128,128
        )
        # 16 224 224
        self.conv2 = nn.Sequential(
            nn.Conv2d( # 1 224 224
                in_channels=32, # input rgb size
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2 # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Conv2d(  # 1 224 224
                in_channels=64,  # input rgb size
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2  # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16,128,128
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d( # 1 224 224
                in_channels=128, # input rgb size
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=2 # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Conv2d(  # 1 224 224
                in_channels=256,  # input rgb size
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=2  # （l - k + 2p）/s + 1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16,128,128
        )

        self.fc1 = nn.Linear(25088, 2048)
        self.out= nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv3(x)
        x = x.view(x.shape[0], -1) # flatten
        # print(x.size(), '进入全连接层前的维度')
        x = self.relu(self.fc1(x))
        x = self.out(x) # 全连接层也需要激活函数，切记
        # x = self.fc1(x)
        # x = self.out(x)
        # x = self.softmax(x)
        return x

if __name__ == '__main__':
    x=torch.randn(1,3,32,32) # (batch-size, rgb_channel_size,length,height)
    net=cnn(100) # 做二分类
    output = net(x)
    print(output.shape) # (batchsize,class_num,len,height)
    print(output)
