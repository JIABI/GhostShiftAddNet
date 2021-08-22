# from models import adder
import torch
import torch.nn as nn
import torch.nn.functional as F
from adder import adder
from adder_slow import adder2d

import math

__all__ = ['resnet20_shiftadd_ghost']


def conv_add(in_planes, out_planes, kernel_size, stride, padding, bias=False, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    add = adder.Adder2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=in_planes, padding=padding, bias=bias, quantize=quantize, weight_bits=weight_bits, quantize_v=quantize_v)
    #add = adder2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    return nn.Sequential(add)


def conv5x5(in_planes, out_planes, kernel_size, stride, padding, bias=False, groups=1, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups, bias=bias)
    return nn.Sequential(shift)


def _make_divisible(v, divisor, width=1, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        #return F.relu6(x + 3.) / 6.
        return F.hardswish()


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=10, dw_size=3, stride=1, relu=True, dilation=1,):
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


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.1, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=F.hardswish, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, kernel_size=1, stride=1, bias=False)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, quantize=False, weight_bits=8, quantize_v='sbm'):
        super(BasicBlock, self).__init__()
        self.conv1 = GhostModule(inplanes, planes, kernel_size=1, stride=1)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU, bias=False, quantize=False, weight_bits=8, quantize_v='sbm'):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""
    expansion = 1

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., downsample=None, quantize=False, weight_bits=8, quantize_v='sbm'):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.downsample = downsample
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=True)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)

        return x

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # use conv as fc layer (addernet)
        self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), # shift
                # adder.adder2d(planes * block.expansion, planes * block.expansion, kernel_size=1, stride=1, bias=False), # add
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)
        return x.view(x.size(0), -1)

class ResNet_ghost(nn.Module):
    def __init__(self, cfgs, num_classes=10, kernel_size=3, width=1.0, dropout=0.1, quantize=False, weight_bits=8, quantize_v='sbm'):
        super(ResNet_ghost, self).__init__()
        self.inplanes = 16
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.quantize_v = quantize_v
        self.dropout = dropout
        self.cfgs = cfgs
        layers = []
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        planes = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, planes, 3, 1, 1, groups= 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        inplanes = planes

        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            downsample = None
            for k, exp_size, c, se_ratio, s in cfg:
                planes = _make_divisible(c * 1, 4)
                hidden_channel = _make_divisible(exp_size * 1, 4)
                layers.append(block(inplanes, hidden_channel, planes, k, s,
                                    se_ratio=se_ratio, downsample=downsample, quantize=self.quantize,
                                    weight_bits=self.weight_bits, quantize_v=self.quantize_v))
                inplanes = planes
                downsample = None
            if s != 1 or inplanes != planes * exp_size:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * 1, kernel_size=1, stride=1, bias=False),
                    # shift
                    #adder.Adder2D(planes * 1, planes * 1, kernel_size=1, stride=1,
                    #              bias=False,
                    #              quantize=self.quantize, weight_bits=self.weight_bits, quantize_v=self.quantize_v),
                    # add
                    nn.BatchNorm2d(planes * exp_size)
                )
            stages.append(nn.Sequential(*layers))
        planes = _make_divisible(exp_size * 1, 4)
        stages.append(nn.Sequential(ConvBnAct(inplanes, planes, 1)))
        inplanes = planes

        self.blocks = nn.Sequential(*stages)

        # use conv as fc layer (addernet)
        #planes = 64
        #self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.conv_head = nn.Linear(inplanes, planes)
        #self.bn = nn.BatchNorm2d(num_classes)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.conv_head = nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(planes, num_classes)

    #def _make_layer(self, block, planes, blocks, stride=1):

    def forward(self, x):
            x = self.conv_stem(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.blocks(x)

            #x = self.global_pool(x)
            x = self.avgpool(x)
            x = self.conv_head(x)
            x = self.bn(x)
            return x.view(x.size(0), -1)
            #if self.dropout > 0.:
            #    x = F.dropout(x, p=self.dropout, training=self.training)
            #x = self.classifier(x)




class ResNet_vis(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_vis, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # use linear as fc layer
        self.linear_1 = nn.Linear(64 * block.expansion, 2)
        self.linear_2 = nn.Linear(2, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), # shift
                adder.Adder2D(planes * block.expansion, planes * block.expansion, kernel_size=1, stride=1, bias=False), # add
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = self.linear_1(x)
        x = self.linear_2(feat)
        return feat, x.view(x.size(0), -1)

def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 32, 16, 0, 1]],
        #[[3, 32, 32, 0, 1]],
        # stage3
        [[3, 64, 32, 0, 2]],
       # [[3, 64, 64, 0, 1]],
        # stage4
          [[3, 64, 64, 0, 1]],
        #  [[3, 64, 64, 0, 1],
           #[3, 64,  64, 0, 1],
           #[3, 64, 64, 0, 1],
           #[3, 64, 64, 0, 1],
          #[3, 256, 112, 0.25, 1]
         #  ],
        # stage5
         #[[3, 128, 112, 0.25, 2]],
         #[[3, 480, 112, 0, 1],
         #[3, 64, 64, 0.25, 1],
        #  [5, 960, 160, 0, 1],
        #  [5, 960, 160, 0.25, 1]
         # ]
    ]
    return ResNet(cfgs, **kwargs)

def resnet20_shift(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


