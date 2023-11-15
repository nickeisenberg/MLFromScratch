import torch
import torch.nn as nn


class CNNBlock(nn.Module): 
	def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs): 
		super().__init__()
		self.use_batch_norm = use_batch_norm 
		self.conv = nn.Conv2d(in_channels, out_channels, bias=(not use_batch_norm), **kwargs) 
		self.bn = nn.BatchNorm2d(out_channels) 
		self.activation = nn.LeakyReLU(0.1) 

	def forward(self, x): 
		x = self.conv(x) 
		if self.use_batch_norm: 
			x = self.bn(x) 
			return self.activation(x) 
		else: 
			return x

class ResBlock(nn.Module):
    def __init__(self, channels, use_residual, num_repeats):
        super().__init__()
        self.use_residual = use_residual
        self.num_repeats = num_repeats

        self._layers = []
        for _ in range(num_repeats):
            self._layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels // 2, kernel_size=1),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(.1),
                    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(.1)
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

class ScalePredictionBlocl(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential( 
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(2 * in_channels), 
            nn.LeakyReLU(0.1), 
            nn.Conv2d(2 * in_channels, (num_classes + 5) * 3, kernel_size=1), 
        ) 
        self.num_classes = num_classes 

    def forward(self, x):
        output = self.pred(x)
        output = output.view(
            x.shape[0], 3, x.shape[2], x.shape[3], self.num_classes + 5
        )
        return output

class Yolo(nn.Module):
    def __init__(self):
        super().__init__()



x = torch.randn(10, 3, 16, 16)

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




