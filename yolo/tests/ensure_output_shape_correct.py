"""
Some helpful links

https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/
https://medium.com/analytics-vidhya/object-detection-state-of-the-art-yolo-v3-79ad2937832
"""

import torch
from model import *


#--------------------------------------------------
# The model produces the correct size output
#--------------------------------------------------
input = torch.randn(1, 1, 256, 256)

# flir shape
input = torch.randn((1, 1, 512, 640), requires_grad=True)

anchors = torch.tensor([ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
]).reshape((-1, 2))
anchors

yoloV3 = YoloV3((1, 512, 640), anchors, [32, 16, 8], 80)

for t in yoloV3(input):
    print(t.shape)

#--------------------------------------------------

#--------------------------------------------------
# Below was the intially testing to ensure the model produced the correct
# shape output
#--------------------------------------------------
c1 = ConvBlock(input.shape[0], 32, kernel_size=3, stride=1, padding=1)(input)
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


