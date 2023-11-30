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
epochs = 50

yoloV3model.fit(
    num_epochs=epochs, 
    save_best_train_model_to=save_best_train_to,
    save_best_val_model_to=save_best_val_to,
    save_train_loss_csv_to=save_train_loss_csv_to,
    save_val_loss_csv_to=save_val_loss_csv_to,
)
