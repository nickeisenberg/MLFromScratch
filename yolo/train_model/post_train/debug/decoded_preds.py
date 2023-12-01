from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
import torch
import torch.nn as nn
from train_model.settings import model_inputs, num_classes, t_at
import os
import matplotlib.pyplot as plt
from model import YoloV3, YoloV3Loss, Model
from utils import scale_anchors, BuildTarget, iou
import torch.nn as nn

#--------------------------------------------------
# Instantate the model and BuildTarget class
#--------------------------------------------------
scales = model_inputs['scales']
anchors = model_inputs['anchors']
cat_map = t_at.cat_mapper
cat_map_inv = {v: k for k, v in cat_map.items()}

bt = BuildTarget(cat_map, anchors, scales, 640, 512)

loss_fn = YoloV3Loss("cpu")

yoloV3 = YoloV3(1, scales, num_classes)

model = Model(model=yoloV3, loss_fn=loss_fn, scales=scales, anchors=anchors)

#--------------------------------------------------

#--------------------------------------------------
# Load model weights
#--------------------------------------------------
modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/train_model"
train_pth_path = modelroot + "/state_dicts/yolo_train0.pth"

yoloV3.load_state_dict(torch.load(train_pth_path))
model.load_state_dict(train_pth_path)
#--------------------------------------------------

#--------------------------------------------------
# Get a training image and target and prediction
#--------------------------------------------------
image, target = model_inputs['t_dataset'][1]
image = image.unsqueeze(0)
target = [t.unsqueeze(0) for t in target]
pred = [p.detach() for p in model.model(image)]
#--------------------------------------------------

#--------------------------------------------------
# See the models loss on this image
#--------------------------------------------------
loss = {}
for scale_id, (p, t) in enumerate(zip(pred, target)):
    scale = scales[scale_id]
    scaled_anchors = scale_anchors(
        anchors[3 * scale_id: 3 * (scale_id + 1)], scale, 640, 512
    )
    total_loss, loss_hist = model.loss_fn(p, t, scaled_anchors)
    for k in loss_hist.keys():
        if k not in loss:
            loss[k] = 0
        loss[k] += loss_hist[k]
for k in loss:
    print(k, loss[k])

#--------------------------------------------------
# loss line by line
#--------------------------------------------------

from copy import deepcopy

_target = deepcopy(target)
_pred = deepcopy(pred)

for scale_id in [0, 1, 2]:
    scale = scales[scale_id]
    scaled_anchors = scale_anchors(
        anchors[3 * scale_id: 3 * (scale_id + 1)], scale, 640, 512
    )
    
    target = _target[scale_id]
    pred = _pred[scale_id]
    
    obj = target[..., 4] == 1
    no_obj = target[..., 4] == 0
    
    torch.nn.functional.binary_cross_entropy_with_logits( 
        (pred[..., 4:5][no_obj]), (target[..., 4:5][no_obj]), 
    )

#--------------------------------------------------

p_thresh = .8
iou_thresh = .8
is_pred = True
tup = pred
if not is_pred:
    iou_thresh = 1

all = []
for scale_id, t in enumerate(pred):
    _all = []
    scale = scales[scale_id]
    scaled_ancs = scale_anchors(
        anchors[3 * scale_id: 3 * (scale_id + 1)], scale, 640, 512
    )
    dims = list(zip(*torch.where(t[..., 4:5] > p_thresh)[:-1]))
    for dim in dims:
        if is_pred:
            x, y, w, h, p = t[dim][: 5]
            cat = torch.argmax(t[dim][:5])
            x, y = (x + dim[2].item()) * scale, (y + dim[1].item()) * scale
            w = torch.exp(w) * scaled_ancs[dim[0]][0] 
            h = torch.exp(h) * scaled_ancs[dim[0]][1] 
            cat = cat_map_inv[cat.item()]
        else:
            x, y, w, h, p, cat = t[dim]
            x, y = (x + dim[2].item()) * scale, (y + dim[1].item()) * scale
            w = w * scale
            h = h * scale
            cat = cat_map_inv[cat.item()]
        _all.append(
            {
                'bbox': [x.item(), y.item(), w.item(), h.item()], 
                'category_id': cat,
                'p_score': p.item(),
                'scale_id': scale_id,
                'index': dim
            }
        )
    all += _all

all[1]['bbox']

target[all[1]["scale_id"]][all[1]["index"]]

#-------------------------------------------------------------------------------
# The model is predicing a box when there shoudnt be
#-------------------------------------------------------------------------------

w = nn.ReLU()(pred[0][..., 2: 3].reshape(-1).detach())
h = nn.ReLU()(pred[0][..., 3: 4].reshape(-1).detach())
p = pred[0][..., 4: 5].reshape(-1).detach()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(w)
ax[0].plot(p)
ax[1].plot(h)
ax[1].plot(p)
plt.show()

#-------------------------------------------------------------------------------
sorted_all = sorted(all, key=lambda x: x['p_score'], reverse=True)

keep = []
while sorted_all:
    keep.append(sorted_all[0])
    _ = sorted_all.pop(0)
    for i, info in enumerate(sorted_all):
        score = iou(torch.tensor(keep[-1]['bbox']), torch.tensor(info['bbox']))
        if score > iou_thresh:
            sorted_all.pop(i)

