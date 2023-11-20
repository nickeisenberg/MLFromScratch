"""
build and debug the loss function
"""
import torch
import torch.nn as nn
from model import YoloV3Loss, YoloV3LossOld
from utils import iou

# preds
torch.manual_seed(1)
x = torch.randn(4) + 4
y = torch.randn(4) + 4

pred = torch.cat(
    (
        torch.hstack((
            x,
            torch.tensor([0]),
            torch.randn((80,))
        )).repeat((32, 3, 16, 20, 1)),
        torch.hstack((
            y,
            torch.tensor([1]),
            torch.randn((80,))
        )).repeat((32, 3, 16, 20, 1))
    ), 
    dim=0
)
target = torch.cat(
    (
        torch.hstack((
            x,
            torch.tensor([0]),
            torch.randint(0, 10, (1,))
        )).repeat((32, 3, 16, 20, 1)),
        torch.hstack((
            y,
            torch.tensor([1]),
            torch.randint(0, 10, (1,))
        )).repeat((32, 3, 16, 20, 1))
    ), 
    dim=0
)
torch.manual_seed(1)
anchors = torch.randn((3, 2)) + 5

print(pred.shape)
print(target.shape)
print(anchors.shape)

loss_fn = YoloV3Loss()

loss_fn2 = YoloV3LossOld()

loss_fn(pred, target, anchors)

loss_fn2(pred, target, anchors)

loss_fn.history





