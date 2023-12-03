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
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.pre_pred = nn.Sequential(
            ConvBlock(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0), 
            ConvBlock(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1), 
            ConvBlock(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0), 
        )
        self.pred = nn.Sequential(
            ConvBlock(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(out_channels, (num_classes + 5) * 3, kernel_size=1)
        )
        self.num_classes = num_classes
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pre_pred = self.pre_pred(x)
        pred = self.pred(pre_pred)
        pred = pred.view(
            x.shape[0], 3, x.shape[2], x.shape[3], self.num_classes + 5
        )
        return pre_pred, pred

class Concatenater(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsampler = nn.Sequential(
            ConvBlock(
                in_channels, 
                in_channels // 2, 
                kernel_size=1, 
                stride=1, 
                padding=0
            ),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        up = self.upsampler(x[0])
        return torch.cat((up, x[1]), dim=1)

class YoloV3(nn.Module):
    def __init__(self, img_channels, scales, num_classes):
        super().__init__()
        
        self.img_channels = img_channels
        self.scales = scales
        self.num_classes = num_classes
        
        self.block0 = nn.Sequential(
            ConvBlock(self.img_channels, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResBlock(64),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResBlock(128, num_repeats=2)
        )
        
        self.scale3 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ResBlock(256, num_repeats=8)
        )
        
        self.scale2 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ResBlock(512, num_repeats=8)
        )
       
        self.scale1 = nn.Sequential(
            ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ResBlock(1024, num_repeats=4)
        )
        
        self.pred1 = ScalePredictionBlock(
            1024, 1024, self.num_classes
        )
        
        self.pred2 = nn.Sequential(
            Concatenater(512),
            ScalePredictionBlock(768, 512, self.num_classes)
        )
        
        self.pred3 = nn.Sequential(
            Concatenater(256),
            ScalePredictionBlock(384, 256, self.num_classes)
        )

    def forward(self, x):
        x = self.block0(x)
        scale3 = self.scale3(x)
        scale2 = self.scale2(scale3)
        scale1 = self.scale1(scale2)
        pp1, p1 = self.pred1(scale1)
        pp2, p2 = self.pred2((pp1, scale2))
        _, p3 = self.pred3((pp2, scale3))
        return (p1, p2, p3)
