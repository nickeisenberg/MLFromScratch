import pandas as pd
import matplotlib.pyplot as plt

t_path = "/home/nicholas/GitRepos/ml_arcs/yolo/train_model/lossdfs/train1.csv"
t_loss = pd.read_csv(t_path, index_col=0)

fig, ax = plt.subplots(1, 5, figsize=(12, 5))
for i, col in enumerate(t_loss.columns):
    ax[i].plot(t_loss[col].values)
    ax[i].set_title(f"{col}")
plt.show()

v_path = "/home/nicholas/GitRepos/ml_arcs/yolo/train_model/lossdfs/val1.csv"
v_loss = pd.read_csv(v_path, index_col=0)

fig, ax = plt.subplots(1, 5, figsize=(12, 5))
for i, col in enumerate(v_loss.columns):
    ax[i].plot(v_loss[col].values)
    ax[i].set_title(f"{col}")
plt.show()
