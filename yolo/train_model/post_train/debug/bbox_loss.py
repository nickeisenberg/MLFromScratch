"""
The model produces horrible bounding boxes. The bounding box loss and object
loss seem weird too. Both object loss and box loss seem to never decrease.
From the first epoch, both of these losses are really small and they dont
shrink much throughout training. As for the bounding box loss, this has to be
wrong because the bounding boxes are so far off from their actual target.
"""

import torch
import torch.nn as nn
from train_model.settings import model_inputs, num_classes, t_at
import os
from model import YoloV3
from utils import BuildTarget, scale_anchors, iou
from copy import deepcopy
import matplotlib.pyplot as plt

#--------------------------------------------------
# Instantate the model and BuildTarget class
#--------------------------------------------------
scales = model_inputs['scales']
anchors = model_inputs['anchors']
cat_map = t_at.cat_mapper

bt = BuildTarget(cat_map, anchors, scales, 640, 512)

yoloV3 = YoloV3(1, scales, num_classes)

modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/train_model"
train_pth_path = modelroot + "/state_dicts/yolo_train1.pth"

yoloV3.load_state_dict(torch.load(train_pth_path))
#--------------------------------------------------

#--------------------------------------------------
# Get a training image and target and prediction
#--------------------------------------------------
image, target = model_inputs['t_dataset'][0]

image = image.unsqueeze(0)
pred = [p.detach() for p in yoloV3(image)]
target = [t.unsqueeze(0) for t in target]

mse_s = nn.MSELoss(reduction="sum") 
mse_m = nn.MSELoss(reduction="mean") 
bce_s = nn.BCELoss(reduction='sum') 
bce_m = nn.BCELoss(reduction='mean') 

_target, _pred = deepcopy(target), deepcopy(pred)


scale_id = 1

target, pred = _target[scale_id], _pred[scale_id]

scale = scales[scale_id]
scaled_anchors = scale_anchors(
    anchors[scale_id * 3: (scale_id + 1) * 3], 
    scale, 
    640, 512
)
scaled_anchors = scaled_anchors.reshape((1, 3, 1, 1, 2))


obj = target[..., 4] == 1
no_obj = target[..., 4] == 0

bce_m( 
    (pred[..., 4:5][no_obj]), (target[..., 4:5][no_obj]), 
)

box_preds = torch.cat(
    [
        pred[..., 0: 2], 
        torch.exp(pred[..., 2: 4]) * scaled_anchors
    ],
    dim=-1
) 

ious = iou(box_preds[obj], target[..., 0: 4][obj]).detach() 

mse_s(
    pred[..., 4: 5][obj], 
    ious * target[..., 4: 5][obj]
) 

target[..., 2: 4] = torch.log(1e-6 + target[..., 2: 4] / scaled_anchors) 

mse_s(
    pred[..., 0: 4][obj], 
    target[..., 0: 4][obj]
)
