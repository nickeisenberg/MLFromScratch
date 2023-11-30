from model import YoloV3
from train_model.settings import anchors, scales, num_classes
import torch

yoloV3 = YoloV3(1, scales, num_classes)
imgs =  torch.randn(10, 1, 512, 640)
pred = yoloV3(imgs)

for p in pred:
    print(p.shape)

for p in pred:
    probs = p[..., 4: 5]
    x_off = p[..., : 1]
    y_off = p[..., 1: 2]
    p_check = (probs > 1).sum() + (probs < 0).sum()
    x_check = (x_off > 1).sum() + (x_off < 0).sum()
    y_check = (y_off > 1).sum() + (y_off < 0).sum()
    if p_check != 0 or x_check != 0 or y_check != 0:
        print("problem")
    else:
        print("all good")
