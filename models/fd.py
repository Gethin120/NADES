import torch
from torch import nn
import torch.nn.functional as F


class FraudDetection(nn.Module):

    def __init__(self, in_channels, out_channels,dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.bn = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(self.in_channels, int(self.in_channels*2/3))
        self.linear_2 = nn.Linear(int(self.in_channels*2/3),int(self.in_channels*1/3))
        self.linear_3 = nn.Linear(int(self.in_channels*1/3), self.out_channels)

    def reset_parameters(self):
        self.linear_1.reset_parameters()
        self.linear_2.reset_parameters()
        self.linear_3.reset_parameters()

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        # x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(self.linear_2(x))
        x = self.dropout(x)
        x = self.linear_3(x)
        # x = self.relu(self.linear_1(x))
        # x = self.dropout(x)
        # x = self.linear_2(x)

        return x

#