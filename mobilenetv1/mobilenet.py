import torch
import torch.nn as nn

def MobileNetConvBlock(in_channels, out_channels, stride):
    dwblock = nn.Sequential(
        # Depthwise convolution
        nn.Conv2d(
            in_channels, 
            in_channels, 
            3, 
            stride=stride, 
            padding=1, 
            groups=in_channels, 
            bias=False
        ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        # Pointwise convolution
        nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

    return dwblock


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            MobileNetConvBlock(32, 64, 1),
            MobileNetConvBlock(64, 128, 2),
            MobileNetConvBlock(128, 128, 1),
            MobileNetConvBlock(128, 256, 2),
            MobileNetConvBlock(256, 256, 1),
            MobileNetConvBlock(256, 512, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


model = MobileNet(num_classes=10)

x = torch.randn(5, 3, 32, 32)

out = model(x)

print(out.shape)
