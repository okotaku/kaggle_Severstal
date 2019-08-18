import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, use_elu=False, **batchnorm_params):

        super().__init__()

        if use_elu:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride=stride, padding=padding, bias=not (use_batchnorm)),
                nn.ELU(True),
            ]
        else:
            layers = [
              nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, bias=not (use_batchnorm)),
              nn.ReLU(inplace=True),
            ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x

        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
