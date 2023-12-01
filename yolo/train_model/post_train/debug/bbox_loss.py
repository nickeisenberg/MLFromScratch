"""
The model produces horrible bounding boxes. The bounding box loss and object
loss seem weird too. Both object loss and box loss seem to never decrease.
From the first epoch, both of these losses are really small and they dont
shrink much throughout training. As for the bounding box loss, this has to be
wrong because the bounding boxes are so far off from their actual target.
"""

import torch
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
target = [t.unsqueeze(0) for t in target]
pred = [p.detach() for p in yoloV3(image)]

_target, _pred = deepcopy(target), deepcopy(pred)

scale_id = 1
scale = scales[scale_id]
target, pred = _target[scale_id], _pred[scale_id]
scaled_anchors = scale_anchors(
    anchors[scale_id * 3: (scale_id + 1) * 3], 
    scale, 
    640, 512
)

obj = target[..., 4] == 1
no_obj = target[..., 4] == 0

tar_dims = list(zip(*torch.where(obj == True)))
pred_dims = list(zip(*torch.where(pred[..., 4:5] > .65)[:-1]))

len(tar_dims)
len(pred_dims)

pred[0].shape


p_thresh = .65
is_pred=True
all = {0: [], 1: [], 2: []}
for scale_id, t in enumerate(pred):
    t = t[0]
    _all = []
    scale = scales[scale_id]
    scaled_ancs = scale_anchors(
        anchors[3 * scale_id: 3 * (scale_id + 1)], scale, 640, 512
    )
    dims = list(zip(*torch.where(t[..., 4:5] > p_thresh)[:-1]))
    for dim in dims:
        if is_pred:
            x, y, w, h, p = t[dim][: 5]
            x, y = (x + dim[2].item()) * scale, (y + dim[1].item()) * scale
            w = torch.exp(w) * scaled_ancs[dim[0]][0] * scale 
            h = torch.exp(h) * scaled_ancs[dim[0]][1] * scale
        else:
            x, y, w, h, p, cat = t[dim]
            x, y = (x + dim[2].item()) * scale, (y + dim[1].item()) * scale
            w = w * scale
            h = h * scale
        all[scale_id].append(
            {
                'bbox': [x.item(), y.item(), w.item(), h.item()], 
                'p_score': p.item(),
                'index': dim
            }
        )

for dic in all[1]:
    t_bbox = target[1][0][dic['index']][:4]
    print(' ')
    print(dic['bbox'])
    print(t_bbox)
    print(' ')






