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
            ConvBlock(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2)
        )
    def forward(self, x):
        up = self.upsampler(x[0])
        return torch.cat((up, x[1]), dim=1)

class YoloV3(nn.Module):
    def __init__(self, image_size, num_classes):
        super().__init__()

        self.img_w, self.img_h = image_size
        self.num_classes = num_classes
        
        self.block0 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1),
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
        
        self.pred1 = ScalePredictionBlock(1024, 1024, self.num_classes)
        
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

input = torch.randn(1, 3, 256, 256)
input.shape

yoloV3 = YoloV3((416, 416), 10)
yoloV3(input)[2].shape

c1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)(input)
c2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)(c1)
r1 = ResBlock(64)(c2)
c1.shape
c2.shape
r1.shape

c3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)(r1)
r2 = ResBlock(128, num_repeats=2)(c3)
c3.shape
r2.shape

c4 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)(r2)
scale3 = ResBlock(256, num_repeats=8)(c4)
c4.shape
scale3.shape

c5 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)(scale3)
scale2 = ResBlock(512, num_repeats=8)(c5)
c5.shape
scale2.shape

c6 = ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1)(scale2)
scale1 = ResBlock(1024, num_repeats=4)(c6)
c6.shape
scale1.shape

scale1_pp, scale1_p = ScalePredictionBlock(1024, 1024, 10)(scale1)
scale1_pp.shape
scale1_p.shape

u_s1 = Concatenater(512)(scale1_pp, scale2)
u_s1.shape

scale2_pp, scale2_p = ScalePredictionBlock(768, 512, 10)(u_s1)
scale2_pp.shape
scale2_p.shape

u_s2 = Concatenater(256)(scale2_pp, scale3)
u_s2.shape

scale3_pp, scale3_p = ScalePredictionBlock(384, 256, 10)(u_s2)
scale3_pp.shape
scale3_p.shape


