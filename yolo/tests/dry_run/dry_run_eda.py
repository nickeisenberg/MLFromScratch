from tests.dry_run.dry_run_settings import t_dataset, t_at, anchors, scales, num_classes
from model import Model, YoloV3
import matplotlib.pyplot as plt
import os
from utils import BuildTarget, iou
import pandas as pd
import torch

bt = BuildTarget(t_at.cat_mapper, anchors, scales, 640, 512)
yoloV3 = YoloV3(1, num_classes)

yoloV3model = Model(model=yoloV3, scales=scales, anchors=anchors)

modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/tests/dry_run"
save_best_train_to = modelroot + "/state_dicts/yolo_train.pth"
yoloV3model.load_state_dict(save_best_train_to)

#--------------------------------------------------
# load image
#--------------------------------------------------
img, tar = t_dataset[0]
img = img.unsqueeze(0)

#--------------------------------------------------
# make prediction
#--------------------------------------------------
pred = yoloV3model.model(img)

pred = [p[0] for p in pred]

#--------------------------------------------------
# use the decoder
#--------------------------------------------------
t_recover = bt.decode_tuple(tar, .5, 1, False)

p_recover = bt.decode_tuple(pred, .8, .001, True)
        
#--------------------------------------------------
# view the losses
#--------------------------------------------------

train_path = "/home/nicholas/GitRepos/ml_arcs/yolo/tests/dry_run/lossdfs/train.csv"
val_path = "/home/nicholas/GitRepos/ml_arcs/yolo/tests/dry_run/lossdfs/val.csv"

train_loss = pd.read_csv(train_path, index_col=0)
val_loss = pd.read_csv(val_path, index_col=0)

fig, ax = plt.subplots(1, 5, figsize=(12, 5))
for i, col in enumerate(train_loss.columns):
    ax[i].plot(train_loss[col])
    ax[i].set_title(col)
plt.show()
