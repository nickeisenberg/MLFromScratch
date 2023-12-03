from tests.dry_run.dry_run_settings import model_inputs
from model import Model, YoloV3Loss
from utils import scale_anchors
import os

device = model_inputs['device']

anchors = model_inputs['anchors'].to(device)
scales = model_inputs['scales'].to(device)

yoloV3model = Model(**model_inputs)

loss_fn = yoloV3model.loss_fn

modelroot = f"{os.environ['HOME']}/GitRepos/ml_arcs/yolo/tests/dry_run"
save_best_train_to = modelroot + "/state_dicts/yolo_train.pth"
yoloV3model.load_state_dict(save_best_train_to)

for imgs, targets in yoloV3model.t_dataloader:
    imgs = imgs.to(device)
    targets = [target.to(device) for target in targets]
    predictions = yoloV3model.model(imgs)
    for scale_id, (preds, targs) in enumerate(zip(predictions, targets)):
        scaled_anchors = scale_anchors(
            anchors[scale_id * 3: (scale_id + 1) * 3], 
            scales[scale_id],
            640, 512,
            device=device
        )
        _batch_loss, batch_history = loss_fn(
            preds,
            targs,
            scaled_anchors
        )
        print(batch_history)
