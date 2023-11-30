import torch
import torch.nn as nn
from train_model.settings import model_inputs, num_classes, t_annotations
import os
import matplotlib.pyplot as plt
from model import YoloV3, YoloV3Loss
from utils import scale_anchors

scales = model_inputs['scales']
anchors = model_inputs['anchors']

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

tup = target
p_thresh = [.9, .9, .9]
iou_thresh = [.9, .9, .9]
count = 0
for scale_id, (t, pt, it) in enumerate(zip(tup, p_thresh, iou_thresh)):
    dims = list(zip(*torch.where(t[..., 4:5] > p_thresh[scale_id])[:-1]))
    for dim in dims:
        print(dim)
        count += 1
print(count)



