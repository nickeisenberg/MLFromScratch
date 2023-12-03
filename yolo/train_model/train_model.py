from train_model.settings import model_inputs
from model import Model
import os

yoloV3model = Model(**model_inputs)

modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/train_model"
save_best_train_to = modelroot + "/state_dicts/yolo_train3.pth"
save_best_val_to = modelroot + "/state_dicts/yolo_val3.pth"
save_train_loss_csv_to = modelroot + "/lossdfs/train3.csv"
save_val_loss_csv_to = modelroot + "/lossdfs/val3.csv"
epochs = 100

yoloV3model.fit(
    num_epochs=epochs, 
    save_best_train_model_to=save_best_train_to,
    save_best_val_model_to=save_best_val_to,
    save_train_loss_csv_to=save_train_loss_csv_to,
    save_val_loss_csv_to=save_val_loss_csv_to,
)
