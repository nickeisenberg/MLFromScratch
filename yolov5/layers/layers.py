import torch.nn as nn
import torch

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, use_bn=True, act=nn.SiLU(), **kwargs):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding, bias=False, **kwargs)
        self.act = act if act is not None else nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) if self.use_bn else self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True):
        super().__init__()
        _c = out_channels // 2
        self.conv1 = Conv(in_channels, _c, 1, 1, 0)
        self.conv2 = Conv(_c, out_channels, 3, 1, 1)
        self.residual = residual and in_channels == out_channels

    def forward(self, input):
        x = self.conv2(self.conv1(input))
        return input + x if self.residual else x


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        _c = in_channels // 2
        self.conv1 = Conv(in_channels, _c, 1, 1, 0)
        self.conv2 = Conv(_c * 4, out_channels, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size, 1, kernel_size // 2)

    def forward(self, input):
        x = [self.conv1(input)]
        x.extend(self.m(x[-1]) for _ in range(3))
        return self.conv2(torch.cat(x, 1))


class C3(nn.Module):
    def __init__(self, in_channels, out_channels, num_repeats, residual=True):
        super().__init__()
        _c = out_channels // 2
        self.conv1 = Conv(in_channels, _c, 1, 1, 0)
        self.conv2 = Conv(in_channels, _c, 1, 1, 0)
        self.conv3 = Conv(2 * _c, out_channels, 1, 1, 0)
        self.b = nn.Sequential(
            *[Bottleneck(_c, _c, residual) for _ in range(num_repeats)]
        )

    def forward(self, x):
        return self.conv3(torch.cat((self.b(self.conv1(x)), self.conv2(x)), 1))


