import pandas as pd
from sshtools.transfer import scp
import os

modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/train_model"
save_best_train_to = modelroot + "/state_dicts/yolo_train.pth"
save_best_val_to = modelroot + "/state_dicts/yolo_val.pth"
save_train_loss_csv_to = modelroot + "/lossdfs/train.csv"
save_val_loss_csv_to = modelroot + "/lossdfs/val.csv"

src = f"/home/ubuntu/GitRepos/ml_arcs/yolo/train_model/state_dicts/yolo_train.pth"
dst = f"/home/nicholas/GitRepos/ml_arcs/yolo/train_model/state_dicts/yolo_train0.pth"
user = "nicholas"
ip = "174.72.155.21"
port = "2201"

scp(src, dst, user, ip, port)
