from tests.train_model.settings import *
from model import Model
import matplotlib.pyplot as plt
import numpy as np
from sshtools.plotting import Plotter

yoloV3model = Model(
    yoloV3, loss_fn, optimizer, t_dataset, v_dataset,
    batch_size, device, scales, anchors, notify_after
)

yoloV3model.device

yoloV3model.fit(num_epochs=epochs)

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
        yoloV3model.history[key][i] for i in range(1, len(yoloV3model.history[key]) + 1)
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
save_path = "/home/nicholas/GitRepos/ml_arcs/yolo/tests/train_model/lossplots"
user = "nicholas"
ip = "174.72.155.21"
plotter = Plotter(user, ip, save_path, port)

plotter.show("val_loss", vallossfig)

plotter.show("train_loss", trainlossfig)
