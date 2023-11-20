"""
build and debug the loss function
"""
import torch
import torch.nn as nn
from model import YoloV3Loss
from utils import iou


# preds
pred = torch.cat(
    (
        torch.hstack((
            torch.randn(4) + 4,
            torch.Tensor([0]),
            torch.randn((80,))
        )).repeat((32, 3, 16, 20, 1)),
        torch.hstack((
            torch.randn(4) + 4,
            torch.Tensor([1]),
            torch.randn((80,))
        )).repeat((32, 3, 16, 20, 1))
    ), 
    dim=0
)
target = torch.cat(
    (
        torch.hstack((
            torch.randn(4) + 4,
            torch.Tensor([0]),
            torch.randint(0, 10, (1,))
        )).repeat((32, 3, 16, 20, 1)),
        torch.hstack((
            torch.randn(4) + 4,
            torch.Tensor([1]),
            torch.randint(0, 10, (1,))
        )).repeat((32, 3, 16, 20, 1))
    ), 
    dim=0
)
anchors = torch.randn((3, 2)) + 5
anchors = anchors.reshape((1, 3, 1, 1, 2))

print(pred.shape)
print(target.shape)
print(anchors.shape)

loss_fn = YoloV3Loss()

loss_fn(pred, target, anchors)

loss_fn.history
