"""
The unet architecture.
Basic Diagram 

inputs             outputs
  |                    |
(s1, p1) --(s1)-----> d1
  |(p1) ----b1------>  |

Idea:
* s1 is a channel increase of inputs
* p1 is a channel increase and down sample of height and width of inputs
* b1 is a channel increase of p1. In otherwords, b1 is a channel increase and 
  down sample of height and width of s1
* d1 is a reconstruction with shape s1
* outputs is a reconstruction of inputs, in otherwords, it is a channel decrease
  of d1.
"""


import torch
import torch.nn as nn


class ConvBlock(nn.Module):

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

        self.convblock = ConvBlock(
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
    

class DecoderBlock(nn.Module):

    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            ):
        
        super().__init__()
        
        self.convT = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            stride=2,
            padding=0,
            kernel_size=2
        )

        self.convblock = ConvBlock(2 * out_channels, out_channels)


    def forward(
            self, 
            inputs: torch.Tensor,
            skip: torch.Tensor
            ):

        x = self.convT(inputs)
        x = torch.cat((x, skip), axis=1)
        x = self.convblock(x)

        return x

class UNet(nn.Module):
    
    def __init__(
            self,
            ):

        super().__init__()
        self.e1 = EncoderBlock(3, 16)
        self.b  = ConvBlock(16, 32)
        self.d1 = DecoderBlock(32, 16)
        self.output = ConvBlock(16, 3)

    def forward(
            self,
            inputs: torch.Tensor,
            ):

        s1, p1 = self.e1(inputs)
        b = self.b(p1)
        d1 = self.d1(b, s1)
        out = self.output(d1)

        return out



#--------------------------------------------------
# Testing
#--------------------------------------------------
inputs = torch.randn(size=(16, 3, 64, 64))


cb = ConvBlock(3, 16)
cb_out = cb(inputs)
cb_out.shape

eb = EncoderBlock(3, 16)
eb_out = eb(inputs)
eb_out[0].shape
eb_out[1].shape


b = ConvBlock(16, 32)
b(eb_out[1]).shape

db = DecoderBlock(32, 16)

db(b(eb_out[1]), eb_out[0]).shape

convT = nn.ConvTranspose2d(
    32,
    16,
    stride=2,
    padding=0,
    kernel_size=2
)

convT(b(eb_out[1])).shape

eb_out[0].shape

conv = ConvBlock(32, 16)

conv(torch.cat(
    (convT(b(eb_out[1])),eb_out[0]), axis=1
)).shape

