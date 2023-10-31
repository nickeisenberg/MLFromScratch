import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        
        identity = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.block1 = ConvBlock(64, 128, 2)
        self.block2 = ConvBlock(128, 256, 2)
        self.block3 = ConvBlock(256, 512, 2)
        self.block4 = ConvBlock(512, 512, 1)
        self.block5 = ConvBlock(512, 512, 1)

        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        self.linear = nn.Linear(512, num_classes)

    def forward(self, inputs):

        x = self.initial(inputs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.linear(x)

        return x
