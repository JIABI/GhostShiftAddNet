import torch
from torch import nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
from typing import Callable, Any, Optional, List
import torch.nn.functional as F
from adder import adder
import math
from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

def conv_add(in_planes, out_planes, kernel_size, stride, padding, bias=False, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    add = adder.Adder2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=in_planes, padding=padding, bias=bias, quantize=quantize, weight_bits=weight_bits, quantize_v=quantize_v)
    #add = adder2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    return nn.Sequential(add)

def conv5x5(in_planes, out_planes, kernel_size, stride, padding, bias=False, groups=1, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups, bias=bias)
    return nn.Sequential(shift)

def conv_1x1(in_planes, out_planes, kernel_size, stride, padding, bias=False, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    shift = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=1//2, stride=stride, groups=1,
                      bias=bias)
    add = adder.Adder2D(out_planes, out_planes, kernel_size=1, stride=stride, groups=1, padding=0, bias=bias, quantize=quantize, weight_bits=weight_bits, quantize_v=quantize_v)
    #add = adder2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    return nn.Sequential(shift, add)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=5, dw=1, ratio=2, dw_size=3, stride=1, relu=True, dilation=1,):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        padding = (kernel_size - 1) // 2 * dilation

        self.primary_conv = nn.Sequential(
            # conv_add(inp, init_channels, kernel_size, 1, kernel_size // 2, bias=False),
            conv5x5(inp, init_channels,  kernel_size=3, stride=1, padding=3 // 2, groups=1, bias=False),
            nn.BatchNorm2d(init_channels),
            #nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            conv5x5(init_channels, new_channels, kernel_size=3, stride=1, padding=3 // 2, groups=init_channels, bias=False),
            #nn.BatchNorm2d(new_channels),
            conv_add(new_channels, new_channels, dw, stride=1, padding=1 // 2, bias=False),
            # conv_1x1(init_channels, new_channels, dw_size, 1, dw_size // 2,  bias=False),
            nn.BatchNorm2d(new_channels),
            #nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]



class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            #GhostModule(in_planes, out_planes, kernel_size),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes
ConvBNReLU = ConvBNActivation

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            #layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
            layers.append(GhostModule(inp, hidden_dim, kernel_size=3))
            #layers.append(conv_1x1(inp, hidden_dim, kernel_size=1, stride=1, padding=1 // 2, bias=False))
        layers.extend([
            nn.BatchNorm2d(hidden_dim),
            #nn.ReLU6(inplace=True),
            # pw-linear
            # dw
            #ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            #GhostModule(hidden_dim, hidden_dim, stride=stride),
            #nn.BatchNorm2d(oup),
            #nn.ReLU6(inplace=True)
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=3 // 2),
            #GhostModule(hidden_dim, oup, stride=1, kernel_size=3),
            #nn.BatchNorm2d(oup),
            #nn.Identity(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),

        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float =1,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 1,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [
            ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model