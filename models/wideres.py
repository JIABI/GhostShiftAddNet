import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import math
from adder import adder
from adder_slow import adder2d
import sys
import numpy as np
__all__ = ['wideres']
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
def conv_add(in_planes, out_planes, kernel_size, stride, padding, bias=False, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    add = adder.Adder2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=in_planes, padding=padding, bias=bias, quantize=quantize, weight_bits=weight_bits, quantize_v=quantize_v)
    #add = adder2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    return nn.Sequential(add)

def conv5x5(in_planes, out_planes, kernel_size, stride, padding, bias=False, groups=1, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups, bias=bias)
    return nn.Sequential(shift)

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=0, relu=True, dilation=1,):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        padding = (kernel_size - 1) // 2 * dilation

        self.primary_conv = nn.Sequential(
            conv5x5(inp, init_channels, kernel_size, 1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(init_channels),
 #           conv5x5(init_channels, init_channels, dw_size, 1, 1,  groups=init_channels, bias=False),
 #           nn.BatchNorm2d(init_channels),
            nn.Identity(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            conv_add(init_channels, new_channels, dw_size, 1, padding=1,  bias=True),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = GhostModule(in_planes, planes, kernel_size=1)
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=True)
        #self.conv2 = GhostModule(planes, planes, kernel_size=3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                #GhostModule(in_planes, planes, kernel_size=1),
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True),
            )




    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = GhostModule(3, nStages[0], kernel_size=3, stride=1, padding=1)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

