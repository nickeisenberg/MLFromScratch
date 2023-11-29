from train_model.settings import *
from model import Model
import matplotlib.pyplot as plt
import numpy as np
from sshtools.plotting import Plotter
from sshtools.transfer import scp
import os

import pandas as pd

yoloV3model = Model(**model_inputs)

modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/train_model"
save_model_to = modelroot + "/state_dicts/yolotemp.pth"
save_train_loss_csv_to = modelroot + "/lossdfs/train.csv"
save_val_loss_csv_to = modelroot + "/lossdfs/val.csv"

yoloV3model.fit(
    num_epochs=epochs, 
    save_model_to=save_model_to,
    save_train_loss_csv_to=save_train_loss_csv_to,
    save_val_loss_csv_to=save_val_loss_csv_to,
)

max(1, len(yoloV3model.history['total_loss'][1]) // 5)

len(yoloV3model.history['total_loss'][1][::30])


train_losses = {key: [] for key in yoloV3model.history.keys()}
for key in yoloV3model.history.keys():
    for epoch in range(1, len(yoloV3model.history[key]) + 1):
        by = max(1, len(yoloV3model.history[key][epoch]) // 5)
        train_losses[key] += yoloV3model.history[key][epoch][:: by]
        print(len(train_losses[key]))
        print(type(yoloV3model.history[key][epoch][:: by]))
        # print(len(yoloV3model.history[key][epoch][:: by]))
        if train_losses[key][-1] != yoloV3model.history[key][epoch][-1]:
            train_losses[key].append(
                yoloV3model.history[key][epoch][-1]
            )

len(train_losses['total_loss'])





import pandas as pd
pd.read_csv("/home/nicholas/GitRepos/ml_arcs/yolo/train_model/lossdfs/train.csv").shape
pd.read_csv("/home/nicholas/GitRepos/ml_arcs/yolo/train_model/lossdfs/val.csv").shape

loss_keys = [
    "box_loss",
    "object_loss",
    "no_object_loss",
    "class_loss",
    "total_loss"
]
stacked_losses = {}
for key in loss_keys:
    stacked_losses[key] = np.hstack([
        yoloV3model.history[key][epoch]
        for epoch in range(1, len(yoloV3model.history[key]) + 1)
    ])

x = pd.DataFrame.from_dict(stacked_losses)

#--------------------------------------------------
# view the losses
#--------------------------------------------------
loss_keys = [
    "box_loss",
    "object_loss",
    "no_object_loss",
    "class_loss",
    "total_loss"
]

stacked_losses = {}
for key in loss_keys:
    stacked_losses[key] = np.hstack([
        np.mean(yoloV3model.history[key][epoch])
        for epoch in range(1, len(yoloV3model.history[key]) + 1)
    ])

trainlossfig, ax = plt.subplots(1, 5, figsize=(12, 5))
for i in range(len(ax)):
    ax[i].plot(stacked_losses[loss_keys[i]])
    ax[i].set_title(f"{loss_keys[i]}")

vallossfig, ax = plt.subplots(1, 5, figsize=(12, 5))
for i in range(len(ax)):
    ax[i].plot(yoloV3model.val_history[loss_keys[i]])
    ax[i].set_title(f"{loss_keys[i]}")

port = "2222"
save_path = "/home/nicholas/GitRepos/ml_arcs/yolo/train_model/lossplots"
user = "nicholas"
ip = "174.72.155.21"
plotter = Plotter(user, ip, save_path, port)

plotter.show("val_loss_1_0", vallossfig)

plotter.show("train_loss_1_0", trainlossfig)

#--------------------------------------------------
# move model to local
#--------------------------------------------------
src = "/home/ubuntu/GitRepos/ml_arcs/yolo/train_model/state_dicts/yolo_1.pth"
dst = "/home/nicholas/GitRepos/ml_arcs/yolo/train_model/state_dicts/yolo_1.pth"
usr = "nicholas"
ip = "174.72.155.21"
port = "2222"
scp(src, dst, usr, ip, port)
