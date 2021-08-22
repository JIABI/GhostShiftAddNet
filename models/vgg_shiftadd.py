'''
Modified from https://github.com/pytorch/vision.git
Copy from https://github.com/Jerry-2017/DoubleBlindImage/blob/master/code/gaussiansmooth/vgg.py
'''
import torch
from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence
from adder_slow import adder2d
from torchvision.models.utils import load_state_dict_from_url
from models.mobilenet_v2_backbone import _make_divisible, ConvBNActivation
import math
from adder import adder
# from se_shift import SEConv2d, SELinear

__all__ = [
    "vgg19_small_shiftadd"
]

def conv3x3(in_planes, out_planes, stride=1, quantize=False, weight_bits=8, sparsity=0):
    " 3x3 convolution with padding "
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    add = adder.Adder2D(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity)
    return nn.Sequential(shift, add)

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
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, dilation=1,):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        padding = (kernel_size - 1) // 2 * dilation

        self.primary_conv = nn.Sequential(
            conv5x5(inp, init_channels, kernel_size, 1, kernel_size // 2, groups=1, bias=False),
            nn.BatchNorm2d(init_channels),
            conv5x5(init_channels, init_channels, dw_size, 1, dw_size // 2,  groups=init_channels, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.Identity(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            conv_add(init_channels, new_channels, dw_size, 1, dw_size // 2,  bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10, dropout=True, small=False, supersmall=False):
        super(VGG, self).__init__()
        self.features = features
        cls_layers = []
        if dropout or supersmall:
            cls_layers.append(nn.Dropout())
        if not (small or supersmall):
            cls_layers.append(nn.Linear(512, 512))
            cls_layers.append(nn.ReLU())
            if dropout:
                cls_layers.append(nn.Dropout())
        if not supersmall:
            cls_layers.append(nn.Linear(512, 512))
            cls_layers.append(nn.ReLU())
        cls_layers.append(nn.Linear(512, num_classes))

        self.classifier = nn.Sequential(*cls_layers)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, quantize=False, weight_bits=8, sparsity=0, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = GhostModule(in_channels, v, kernel_size=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))

def vgg11_nd():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), dropout=False)

def vgg11_nd_s():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), dropout=False, small=True)

def vgg11_nd_ss():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), dropout=False, small=True, supersmall=True)


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))

def vgg13_nd():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), dropout=False)

def vgg13_nd_s():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), dropout=False, small=True)

def vgg13_nd_ss():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), dropout=False, small=True, supersmall=True)


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))

def vgg16_nd():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), dropout=False)

def vgg16_nd_s():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), dropout=False, small=True)

def vgg16_nd_ss():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), dropout=False, small=True, supersmall=True)


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))

def vgg19_nd():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), dropout=False)

def vgg19_small_shiftadd(num_classes=10, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'], **kwargs),
               num_classes=num_classes, dropout=False, small=True)

def vgg19_nd_ss():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), dropout=False, small=True, supersmall=True)



def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))