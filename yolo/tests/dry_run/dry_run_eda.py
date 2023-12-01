from tests.dry_run.dry_run_settings import t_dataset, t_at, anchors, scales, num_classes
from model import Model, YoloV3
import matplotlib.pyplot as plt
import os
from utils import BuildTarget, iou
import pandas as pd
import torch

bt = BuildTarget(t_at.cat_mapper, anchors, scales, 640, 512)

yoloV3 = YoloV3(1, scales, num_classes)

yoloV3model = Model(model=yoloV3, scales=scales, anchors=anchors)

modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/tests/dry_run"
save_best_train_to = modelroot + "/state_dicts/yolo_train.pth"
yoloV3model.load_state_dict(save_best_train_to)

img, tar = t_dataset[0]
img = img.unsqueeze(0)

pred = yoloV3model.model(img)

pred = [p[0] for p in pred]

t_recover = bt.decode_tuple(tar, .5, 1, False)

p_recover = bt.decode_tuple(pred, .8, .001, True)
len(p_recover)

for p in p_recover:
    p_bbox = torch.tensor(p['bbox'])
    for pp in p_recover:
        pp_bbox = torch.tensor(pp['bbox'])
        score = iou(p_bbox, pp_bbox)
        if score > 0 and score < .99:
            print(score)
        
#--------------------------------------------------
# view the losses
#--------------------------------------------------

train_loss = pd.read_csv(save_train_loss_csv_to, index_col=0)
val_loss = pd.read_csv(save_val_loss_csv_to, index_col=0)

train_loss['total_loss']

plt.plot(train_loss['total_loss'])
plt.show()
