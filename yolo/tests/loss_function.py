"""
build and debug the loss function
"""
import torch
import torch.nn as nn
from utils import iou


class YoloV3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() 
        self.bce = nn.BCEWithLogitsLoss() 
        self.cross_entropy = nn.CrossEntropyLoss() 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, pred, target, scaled_anchors):

        obj = target[..., 4] == 1
        no_obj = target[..., 4] == 0

        no_object_loss = self.bce( 
            (pred[..., 4:5][no_obj]), (target[..., 4:5][no_obj]), 
        )

        scaled_anchors = scaled_anchors.reshape((1, 3, 1, 1, 2))

        box_preds = torch.cat(
            [
                self.sigmoid(pred[..., 0:2]), 
                torch.exp(pred[..., 2:4]) * scaled_anchors 
            ],
            dim=-1
        ) 

        ious = iou(box_preds[obj], target[..., 0:4][obj]).detach() 

        object_loss = self.bce(
            self.sigmoid(pred[..., 4:5][obj]), 
            ious * target[..., 4:5][obj]
        ) 
       
        # Predicted box coordinates 
        pred[..., 0:2] = self.sigmoid(pred[..., 0:2])

        # Target box coordinates 
        target[..., 2:4] = torch.log(1e-6 + target[..., 2:4] / scaled_anchors) 
        # Calculating box coordinate loss 
        box_loss = self.mse(pred[..., 0:4][obj], 
                            target[..., 0:4][obj]) 
          
        # Claculating class loss 
        class_loss = self.cross_entropy((pred[..., 5:][obj]), 
                                        target[..., 5][obj].long()) 

        # Total loss 
        return ( 
            box_loss 
            + object_loss 
            + no_object_loss 
            + class_loss 
        )


# preds
pred1 = torch.hstack((
    torch.randn(4),
    torch.Tensor([1]),
    torch.randn(80)
))
pred0 = torch.hstack((
    torch.randn(4) + 10,
    torch.Tensor([1]),
    torch.randn(80)
))

# targets
target1 = torch.hstack((
    torch.randn(4),
    torch.Tensor([1]),
    torch.randint(0, 10, (1,))
))
target0 = torch.hstack((
    torch.randn(4) + 10,
    torch.Tensor([1]),
    torch.randint(0, 10, (1,))
))







