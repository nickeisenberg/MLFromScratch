import torch
import torch.nn as nn


class ConvBlockEncoder(nn.Module):

    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            ):
        
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            stride=1,
            padding=1,
            kernel_size=3
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            stride=1,
            padding=1,
            kernel_size=3
        )

    def forward(
            self, 
            inputs: torch.Tensor
            ):

        x = self.conv1(inputs)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)

        return x


class EncoderBlock(nn.Module):

    def __init__(
            self,
            in_channels: int, 
            out_channels: int, 
            ):

        super().__init__()

        self.convblock = ConvBlockEncoder(
                in_channels=in_channels,
                out_channels=out_channels,
                )

        self.pool = nn.MaxPool2d((2, 2))

    def forward(
            self,
            inputs: torch.Tensor,
            ):

        x = self.convblock(inputs)
        p = self.pool(x)

        return x, p
    

class ConvBlockDecoder(nn.Module):

    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            skip: torch.Tensor
            ):
        
        super().__init__()
        
        self.convT = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            stride=2,
            padding=0,
            kernel_size=2
        )

        self.convblock = ConvBlockEncoder(2 * out_channels, out_channels)

        self.skip = skip

    def forward(self, inputs):

        x = self.convT(inputs)
        x = torch.cat((x, self.skip), axis=1)
        x = self.convblock(x)

        return x



#--------------------------------------------------
# Testing
#--------------------------------------------------
input = torch.randn(size=(16, 3, 64, 64))

enc = ConvBlockEncoder(3, 16, if_pooling=True)
output = enc(input)
output.shape


dec = ConvBlockDecoder(16, 3, input)
dec(output).shape
