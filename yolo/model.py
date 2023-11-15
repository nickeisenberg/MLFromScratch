import torch
import torch.nn as nn


class ConvBlock(nn.Module): 
	def __init__(self, in_channels, out_channels, **kwargs): 
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, **kwargs) 
		self.bn = nn.BatchNorm2d(out_channels) 
		self.activation = nn.LeakyReLU(0.1) 

	def forward(self, x): 
		x = self.conv(x) 
		x = self.bn(x) 
		return self.activation(x) 

class ResBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.use_residual = use_residual
        self.num_repeats = num_repeats

        self._layers = []
        for _ in range(num_repeats):
            self._layers.append(
                nn.Sequential(
                    ConvBlock(channels, channels // 2, kernel_size=1),
                    ConvBlock(channels // 2, channels, kernel_size=3, padding=1)
                )
            )
        self.layers = nn.ModuleList(self._layers)

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            if self.use_residual:
                x += residual
            return x

class ScalePredictionBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            ResBlock(in_channels, use_residual=False, num_repeats=3),
            nn.Conv2d(in_channels, (num_classes + 5) * 3, kernel_size=1)
        ) 
        self.num_classes = num_classes 

    def forward(self, x):
        output = self.pred(x)
        output = output.view(
            x.shape[0], 3, x.shape[2], x.shape[3], self.num_classes + 5
        )
        return output

class Upsampler(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsampler = nn.Sequential(
            ConvBlock(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2)
        )
    def forward(self, x):
        return self.upsampler(x)


class Yolo(nn.Module):
    def __init__(self):
        super().__init__()

input = torch.randn(1, 3, 256, 256)

c1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)(input)
c2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)(c1)
r1 = ResBlock(64)(c2)

c3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)(r1)
r2 = ResBlock(128, num_repeats=2)(c3)

c4 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)(r2)
scale3 = ResBlock(256, num_repeats=8)(c4)

c5 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)(scale3)
scale2 = ResBlock(512, num_repeats=8)(c5)

c6 = ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1)(scale2)
scale1 = ResBlock(1024, num_repeats=4)(c6)

scale1_p = ScalePredictionBlock(1024, 10)(scale1)

u1 = Upsampler(1024)(scale1)
pre_scale2_p = torch.cat((scale2, u1), dim=1)

input.shape

c1.shape
c2.shape
r1.shape

c3.shape
r2.shape

c4.shape
scale3.shape

c5.shape
scale2.shape

c6.shape
scale1.shape

scale1_p.shape

u1.shape

pre_scale2_p.shape


x.reshape((-1, 3, -1))

x.view(10, 3 * 16 * 16).shape



nn.Conv2d(3, 6, 3, padding=1)(x).shape

channels = 16
l = [nn.Sequential(
    nn.Conv2d(channels, channels // 2, kernel_size=1),
    nn.BatchNorm2d(channels // 2),
    nn.LeakyReLU(.1),
    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(channels // 2),
    nn.LeakyReLU(.1)
)]

type(l)

for _l in nn.ModuleList(l):
    for __l in _l:
        print(__l)
    break




