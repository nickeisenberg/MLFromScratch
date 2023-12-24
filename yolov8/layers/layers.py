import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, input):
        return self.silu(self.bn(self.conv(input)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.conv1 = Conv(in_channels, in_channels // 2, 3, 1, 1)
        self.conv2 = Conv(in_channels // 2, in_channels, 3, 1, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return x if not self.shortcut else x + input


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_repeats=1, shortcut=False):
        super().__init__()
        self.c = out_channels // 2
        self.conv1 = Conv(in_channels, out_channels, 1, 1, 0)
        self.conv2 = Conv((2 + num_repeats) * self.c, out_channels, 1, 1, 0)
        self.m = nn.ModuleList(
            Bottleneck(self.c, shortcut) 
            for _ in range(num_repeats)
        )

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class Detect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
