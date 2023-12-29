from torch.cuda import is_available
from trfc.experiment import Experiment
import os
from experiment.config import (
    trainer,
    validator,
    train_dataloader,
    val_dataloader,
    yolov5Network,
    logger,
    evaluator
)

save_root = os.path.join(os.environ['HOME'], "GitRepos", "ml_arcs", "yolov5")

experiment = Experiment(
    name="test_run",
    save_root=save_root,
    network=yolov5Network,
    trainer=trainer,
    validator=validator,
    t_dataloader=val_dataloader,
    v_dataloader=val_dataloader,
    logger=logger,
    evaluator=evaluator,
    ts_dataloaders=[val_dataloader],
    num_epochs=1,
    device = "cuda" if is_available() else "cpu"
)

experiment.run()
