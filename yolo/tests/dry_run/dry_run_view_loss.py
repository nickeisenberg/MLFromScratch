import pandas as pd
import matplotlib.pyplot as plt

train_path = "/home/nicholas/GitRepos/ml_arcs/yolo/tests/dry_run/lossdfs/train.csv"
val_path = "/home/nicholas/GitRepos/ml_arcs/yolo/tests/dry_run/lossdfs/val.csv"

train_loss = pd.read_csv(train_path, index_col=0)
val_loss = pd.read_csv(val_path, index_col=0)

fig, ax = plt.subplots(1, 5, figsize=(12, 5))
for i, col in enumerate(train_loss.columns):
    ax[i].plot(train_loss[col])
    ax[i].set_title(col)
plt.show()

fig, ax = plt.subplots(1, 5, figsize=(12, 5))
for i, col in enumerate(val_loss.columns):
    ax[i].plot(val_loss[col])
    ax[i].set_title(col)
plt.show()
