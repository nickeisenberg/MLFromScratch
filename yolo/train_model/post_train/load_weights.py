import torch
import torch.nn as nn
from train_model.settings import model_inputs, num_classes, t_at
import os
import matplotlib.pyplot as plt
from model import YoloV3, YoloV3Loss
from utils import scale_anchors, BuildTarget

scales = model_inputs['scales']
anchors = model_inputs['anchors']
cat_map = t_at.cat_mapper

bt = BuildTarget(cat_map, anchors, scales, 640, 512)

yoloV3 = YoloV3(1, scales, num_classes)

modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/train_model"
train_pth_path = modelroot + "/state_dicts/yolo_train0.pth"

yoloV3.load_state_dict(torch.load(train_pth_path))

image, target = model_inputs['t_dataset'][0]
image = image.unsqueeze(0)
target = [t.unsqueeze(0) for t in target]

plt.plot(target[1][..., 3: 4].detach().reshape(-1))
plt.show()

pred = [p for p in yoloV3(image)]

loss_fn = YoloV3Loss("cpu")
loss = {}
for scale_id, (p, t) in enumerate(zip(pred, target)):
    scale = scales[scale_id]
    scaled_anchors = scale_anchors(
        anchors[3 * scale_id: 3 * (scale_id + 1)], scale, 640, 512
    )
    total_loss, loss_hist = loss_fn(p, t, scaled_anchors)
    for k in loss_hist.keys():
        if k not in loss:
            loss[k] = 0
        loss[k] += loss_hist[k]
for k in loss:
    print(k, loss[k])

len(bt.decode_tuple(target, .8, 1, False))

len(bt.decode_tuple(pred, .75, 1, True))
