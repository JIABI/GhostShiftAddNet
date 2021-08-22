# from models import adder
import torch
import torch.nn as nn
import torch.nn.functional as F
from adder import adder
from torch import Tensor
from typing import Callable, Any, List

#try:
from torch.hub import load_state_dict_from_url
#except ImportError:
#    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import math

__all__ = ['shufflenet_shiftadd']

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def conv3x3(in_planes, out_planes, kernel_size, stride, padding, bias=False, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    add = adder.Adder2D(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, quantize=quantize, weight_bits=weight_bits, quantize_v=quantize_v)
    return nn.Sequential(shift, add)

def conv3x3_DW(in_planes, out_planes, kernel_size, stride, padding, bias=False, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, groups=in_planes, stride=stride, padding=padding, bias=bias)
    add = adder.Adder2D(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, quantize=quantize, weight_bits=weight_bits, quantize_v=quantize_v)
    return nn.Sequential(shift, add)

def conv_add(in_planes, out_planes, kernel_size, stride, padding, bias=False, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    add = adder.Adder2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, quantize=quantize, weight_bits=weight_bits, quantize_v=quantize_v)
    return nn.Sequential(add)

def conv5x5(in_planes, out_planes, kernel_size, stride, padding, groups, bias=False, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, groups=groups)
    return nn.Sequential(shift)

def conv1x1(in_planes, out_planes, stride=1, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm'):
    """1x1 convolution"""
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#    add = adder.Adder2D(out_planes, out_planes, kernel_size=1, stride=1, bias=False, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, quantize_v=quantize_v)
    return nn.Sequential(shift)


def conv1x1sd(in_planes, out_planes, stride=1, quantize=False, weight_bits=8, sparsity=0, quantize_v='sbm'):
    """1x1 convolution"""
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    add = adder.Adder2D(out_planes, out_planes, kernel_size=1, stride=1, bias=False, quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, quantize_v=quantize_v)
    return nn.Sequential(shift, add)

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


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()
#    size_r = x.size()
#    tmp1 = x[:, int(3 * size_r[1] / 4):, :, :]
#    tmp2 = x[:, :int(3 * size_r[1] / 4), :, :]
#    x = torch.cat([tmp1, tmp2], 1)
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, ratio=0.5, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        self.ratio = ratio
        init_channels = math.ceil(oup * self.ratio)
#        new_channels = math.ceil(init_channels * (1-self.ratio))
        new_channels = oup - init_channels

        self.primary_conv = nn.Sequential(
            conv5x5(inp, init_channels, kernel_size, stride, kernel_size//2, groups=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            conv_add(init_channels, new_channels, dw_size, 1, dw_size//2, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostModule_branch1(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule_branch1, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            conv3x3(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            conv5x5(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class InvertedResidual(nn.Module):
    def __init__(self, inp : int, oup : int, stride: int, r1):
        super(InvertedResidual, self).__init__()
 #       if not (1 <= stride <= 3):
 #           raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
#                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp, bias=False),
#                nn.BatchNorm2d(inp),
#                nn.Conv2d(inp, branch_features, stride=1, kernel_size=1, padding=0, bias=False),
#                nn.BatchNorm2d(branch_features),
#                nn.ReLU(inplace=True),
            #    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(inp, branch_features, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch1_1 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
        #    nn.MaxPool2d(kernel_size=3, stride=self.stride, padding=1),
            conv3x3(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            # Squeeze-and-excitation
#            SqueezeExcite(branch_features, se_ratio=0.25) if (self.stride == 1) else nn.Sequential(),

#            conv3x3(branch_features, branch_features, kernel_size=3, stride=self.stride,  padding=1),
#            nn.BatchNorm2d(branch_features),
#            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
#            nn.BatchNorm2d(branch_features),
#            nn.ReLU(inplace=True),

        )
 #       self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out2 = self.branch1_1(x2)
            #out2 = F.max_pool2d(out2, kernel_size=3, stride=self.stride)
            out2 = self.branch2(out2)
            out = torch.cat((x1, out2), dim=1)
        else:
            out1 = F.max_pool2d(x, kernel_size=3, stride=2)
            out1 = self.branch1(out1)
            out2 = self.branch1_1(x)
            out2 = F.max_pool2d(out2, kernel_size=3, stride=2)
            out2 = self.branch2(out2)
            out = torch.cat((out1, out2), dim=1)

        out = channel_shuffle(out, 2)

        return out


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=F.hardswish, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.conv_reduce = conv1x1(in_chs, reduced_chs, stride=1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, kernel_size=1, stride=1, padding=0, bias=False)
        self.act1 = act_layer(inplace=True)
        #self.conv_expand = conv1x1sd(reduced_chs, in_chs, stride=1)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU, bias=False, quantize=False, weight_bits=8, quantize_v='sbm'):
        super(ConvBnAct, self).__init__()
#        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding=kernel_size // 2, groups=in_chs, bias=False)
        self.conv = conv3x3(in_chs, out_chs, kernel_size, stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        kernel_size,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        num_classes: int = 10,
        dropout: float = 0.,
        r1 : float = 0.3,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual
    ) -> None:
        super(ShuffleNetV2, self).__init__()
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        self.dropout =dropout

        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv_shift = nn.Sequential(
        #    conv3x3(input_channels, output_channels, 3, 2, 1, bias=False),
        #    nn.BatchNorm2d(output_channels),
        #    nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, input_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, groups=input_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
 #           conv3x3(output_channels, output_channels, 3, 1, 1, bias=False),
 #           nn.BatchNorm2d(output_channels),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        input_channels = output_channels
        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(inp=input_channels, oup=output_channels, stride=2, r1=0.5)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1, r1=0.5))

            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
#            conv1x1sd(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
#        x = self.conv1(x)
#        x = self.maxpool(x)
#        x = self.conv_shift(x)
#       x = self.maxpool(x)
#        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv1x1(x)
#        x = self.maxpool(x)
#        x = F.max_pool2d(x, kernel_size=3, stride=2)
#        x = self.conv2(x)
#        x = self.maxpool(x)
 #       x = self.conv_shift(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
#        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _shufflenetv2(arch: str, kernel_size, pretrained: bool, progress: bool, *args: Any, **kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2(*args, **kwargs, kernel_size=kernel_size)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)

    return model


#def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
#    cfgs = [
        # k, t, c, SE, s
#        [[3, 32, 24, 0, 1]],
#        [[3, 64, 48, 0, 2]],

#        [[3, 128, 96, 0, 1]],

#        [[3, 256, 192, 0, 1]],
#        [[3, 512, 384, 0, 1]],

#    ]
        # stage1

           #[3, 64, 64, 0, 1],
          #[3, 256, 112, 0.25, 1]
        # stage5
         #[[3, 128, 112, 0.25, 2]],
         #[[3, 480, 112, 0, 1],
         #[3, 64, 64, 0.25, 1],
        #  [5, 960, 160, 0, 1],
        #  [5, 960, 160, 0.25, 1]
         # ]
#    return ResNet(cfgs, **kwargs)


def ghostnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained=False, progress=False, kernel_size=3,
                         stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024], **kwargs)


