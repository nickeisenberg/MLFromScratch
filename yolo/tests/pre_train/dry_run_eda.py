from tests.pre_train.dry_run_settings import model_inputs
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from utils import scale_anchors
import pandas as pd

yoloV3model = Model(**model_inputs)

modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/tests/pre_train"
save_best_train_to = modelroot + "/state_dicts/yolo_train.pth"
save_best_val_to = modelroot + "/state_dicts/yolo_val.pth"
save_train_loss_csv_to = modelroot + "/lossdfs/train.csv"
save_val_loss_csv_to = modelroot + "/lossdfs/val.csv"
epochs = 2

yoloV3model.fit(
    num_epochs=epochs, 
    save_best_train_model_to=save_best_train_to,
    save_best_val_model_to=save_best_val_to,
    save_train_loss_csv_to=save_train_loss_csv_to,
    save_val_loss_csv_to=save_val_loss_csv_to,
)

img, tar = model_inputs["t_dataset"][0]

img = img.unsqueeze(0).to("cuda")

pred = yoloV3model.model(img)

anchors = torch.tensor(model_inputs['anchors']).to('cuda')
scales = torch.tensor(model_inputs['scales']).to('cuda')

for scale_id, (t, p) in enumerate(zip(tar, pred)):
    t = t.to('cuda')
    scaled_ancs = scale_anchors(
        anchors[3 * scale_id: 3 * (scale_id + 1)], 
        scales[scale_id],
        640, 512, 'cuda'
    )
    dims = list(zip(*torch.where(t[..., 4: 5] == 1)[:-1]))
    if len(dims) > 0:
        for dim in dims:
            t_bbox = t[dim][: 5]
            p_bbox = p[0][dim][: 5]
            t_bbox[2: 4] = torch.log(1e-6 + t_bbox[2: 4] / scaled_ancs[dim[0]])
            print("")
            print(p_bbox)
            print(t_bbox)
            print("")

for scale_id, (t, p) in enumerate(zip(tar, pred)):
    t = t.to('cuda')
    scaled_ancs = scale_anchors(
        anchors[3 * scale_id: 3 * (scale_id + 1)], 
        scales[scale_id],
        640, 512, 'cuda'
    )
    dims = list(zip(*torch.where(t[..., 4: 5] == 1)[:-1]))
    if len(dims) > 0:
        for dim in dims:
            t_bbox = t[dim][: 5]
            p_bbox = p[0][dim][: 5].to('cuda')
            p_bbox[2: 4] = torch.exp(p_bbox[2: 4]) * scaled_ancs[dim[0]]
            print("")
            print(p_bbox)
            print(t_bbox)
            print("")

#--------------------------------------------------
# view the losses
#--------------------------------------------------

train_loss = pd.read_csv(save_train_loss_csv_to, index_col=0)
val_loss = pd.read_csv(save_val_loss_csv_to, index_col=0)

train_loss['total_loss']

plt.plot(train_loss['total_loss'])
plt.show()
