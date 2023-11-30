from train_model.settings import model_inputs
from model import Model
import matplotlib.pyplot as plt
import numpy as np
from sshtools.plotting import Plotter
from sshtools.transfer import scp
import os
import torch

yoloV3model = Model(**model_inputs)

modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/train_model"
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
