"""
Nan will appear in the loss. This will happen if either obj.sum() or
no_obj.sum() is 0. If this is the case, then I believe we should set the 
target to be a 0 tensor of the correct shape.
"""

from typing import Tuple
from train_model.settings import t_dataset, scales, anchors
from utils import scale_anchors, iou
import torch
import torch.nn as nn

class YoloV3Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mse = nn.MSELoss() 
        self.bce = nn.BCEWithLogitsLoss() 
        self.cross_entropy = nn.CrossEntropyLoss() 
        self.sigmoid = nn.Sigmoid() 
        self.device = device

    def forward(self, pred, target, scaled_anchors) -> Tuple[torch.Tensor, dict]:
        """
        Recall that the pred and target is a tuple of 3 tensors. As of now,
        This forward only handles each piece separately, ie, pred[0] and 
        target[0] etc. I may generalize to just accept the whole tuple later.

        pred: torch.Tensor, shape=(batch, 3, num_rows, num_cols, 1, 5 + num_classes)
            pred[..., :] = (x, y, w, h, prob, class_probabilities)
        target: torch.Tensor, shape=(batch, 3, num_rows, num_cols, 1, 6)
            target[..., :] = (x, y, w, h, prob, class_ID)

        (x_off, y_off, w/ s, h /s, prop, ....) ---> (x, y, w, h, prob, class)

        (t1, t2, t3) tk.shape = (batch, 3, row, col, ____)
        """

        obj = target[..., 4] == 1
        no_obj = target[..., 4] == 0

        scaled_anchors = scaled_anchors.reshape((1, 3, 1, 1, 2))

        #-------------------------------------------------- 
        # no_obj calculations
        #-------------------------------------------------- 
        no_object_loss = self.bce( 
            (pred[..., 4:5][no_obj]), (target[..., 4:5][no_obj]), 
        )
        #-------------------------------------------------- 

        pred[..., 0: 2] = self.sigmoid(pred[..., 0: 2])
        target[..., 2: 4] = torch.log(1e-6 + target[..., 2: 4] / scaled_anchors) 

        box_preds = torch.cat(
            [
                pred[..., 0: 2], 
                torch.exp(pred[..., 2: 4]) * scaled_anchors
            ],
            dim=-1
        ) 

        #-------------------------------------------------- 
        # obj calculations
        #-------------------------------------------------- 
        ious = iou(box_preds[obj], target[..., 0: 4][obj]).detach() 
        
        object_loss = self.mse(
            self.sigmoid(pred[..., 4: 5][obj]), 
            ious * target[..., 4: 5][obj]
        ) 

        # Calculating box coordinate loss 
        box_loss = self.mse(
            pred[..., 0: 4][obj], 
            target[..., 0: 4][obj]
        )

        # Claculating class loss 
        class_loss = self.cross_entropy(
            pred[..., 5:][obj], 
            target[..., 5][obj].long()
        )
        #-------------------------------------------------- 

        total_loss = box_loss + object_loss + no_object_loss + class_loss 
        
        history = {}
        history["box_loss"] = box_loss.item()
        history["object_loss"] = object_loss.item()
        history["no_object_loss"] = no_object_loss.item()
        history["class_loss"] = class_loss.item()
        history["total_loss"] = total_loss.item()

        return total_loss, history

pred = []
target = []
for _target in t_dataset.__getitem__(1)[1]:
    target.append(_target.unsqueeze(0))
    pred.append(torch.randn(size=_target.shape).unsqueeze(0))

loss_fcn = YoloV3Loss(device='cpu')

losses = []
for i, scale in enumerate(scales):
    scaled_anchors = scale_anchors(anchors[i: i + 3], scales[i], 640, 512)
    loss = loss_fcn(target[i], pred[i], scaled_anchors)
    losses.append(loss)

for loss in losses:
    print(loss)

for tar in target:
    obj = tar[..., 4] == 1
    no_obj = tar[..., 4] == 0
    print(obj.sum(), no_obj.sum())
