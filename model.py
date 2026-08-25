from dataset import *
import torch
import torch.nn as nn

class Classif(nn.Module):
    def __init__(self):
        super(Classif, self).__init__()

        #MLP model 

        #CNN model
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1) #out : 16x32x32
        self.pool1 = nn.MaxPool2d(2, 2) #out : 16x16x16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1) #out : 32x16x16
        self.pool2 = nn.MaxPool2d(2, 2) #out : 32x8x8 
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1) #out : 64x8x8
        self.pool3 = nn.MaxPool2d(2, 2) #out : 64x4x4

        self.relu = nn.ReLU()

        self.last = nn.Linear(in_features = (64 * 4 * 4), out_features=10)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.last(x.view(x.size(0), -1))

        return x
    
    