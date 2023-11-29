from types import NoneType
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import numpy as np
from utils import scale_anchors

class Model:
    def __init__(
        self, 
        model: nn.Module | NoneType = None,
        save_model_to: str | NoneType = None,
        loss_fn: nn.Module | NoneType = None, 
        optimizer: Optimizer | NoneType = None, 
        t_dataset: Dataset | NoneType = None,
        v_dataset: Dataset | NoneType = None,
        batch_size: int | NoneType = None,
        device: str = "cpu",
        scales: torch.Tensor | NoneType = None,
        anchors: torch.Tensor | NoneType = None,
        notify_after: int = 40
        ):
        self.model = model
        self.save_model_to = save_model_to
        if isinstance(loss_fn, nn.Module):
            self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.t_dataset = t_dataset
        self.v_dataset = v_dataset
        if self.t_dataset is not None:
            _, self.img_height, self.img_width = self.t_dataset.__getitem__(0)[0].shape
            self.t_dataloader = DataLoader(
                self.t_dataset, batch_size, shuffle=True
            )
        if self.v_dataset is not None:
            self.v_dataloader = DataLoader(
                self.v_dataset, batch_size, shuffle=False
            )
        self.device = device
        self.history = {
            "box_loss": {},
            "object_loss": {},
            "no_object_loss": {},
            "class_loss": {},
            "total_loss": {},
        }
        self.val_history = {
            "box_loss": [],
            "object_loss": [],
            "no_object_loss": [],
            "class_loss": [],
            "total_loss": [],
        }
        if scales is not None:
            self.scales = scales.to(device)
        if anchors is not None:
            self.anchors = anchors.to(device)
        self.notify_after = notify_after

    def _train_one_epoch(self, epoch):
        assert self.optimizer is torch.optim.Optimizer
        assert self.model is nn.Module

        for key in self.history.keys():
            self.history[key][epoch] = []

        for batch_num, (images, targets) in enumerate(self.t_dataloader):

            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]

            if (batch_num + 1) % self.notify_after == 0:

                batch_loss = np.round(np.mean(self.history['total_loss'][epoch]), 3)
                print(f"EPOCH {epoch} BATCH {batch_num + 1} LOSS {batch_loss}")

            self.optimizer.zero_grad()

            predicitons = self.model(images)
            
            batch_loss = torch.zeros(1, requires_grad=True).to(self.device)
            for scale_id, (preds, targs) in enumerate(zip(predicitons, targets)):
                
                scaled_anchors = scale_anchors(
                    self.anchors[scale_id: scale_id + 3], 
                    self.scales[scale_id],
                    self.img_width, self.img_height,
                    device=self.device
                )

                _batch_loss, batch_history = self.loss_fn(
                    preds,
                    targs,
                    scaled_anchors
                )

                batch_loss += _batch_loss

                for key in self.history.keys():
                    self.history[key][epoch].append(batch_history[key])

            batch_loss.backward()

            self.optimizer.step()
        
        avg_epoch_loss = np.mean(self.history['total_loss'][epoch])
        print(f"EPOCH {epoch} AVG LOSS {np.round(avg_epoch_loss, 3)}")

        return None

    def _validate_one_epoch(self, epoch):
        assert self.model is nn.Module
        assert self.v_dataset is Dataset

        val_epoch_history = {
            "box_loss": 0.,
            "object_loss": 0.,
            "no_object_loss": 0.,
            "class_loss": 0.,
            "total_loss": 0.
        }
        for images, targets in self.v_dataloader:

            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]

            with torch.no_grad():
                predicitons = self.model(images)

                for scale_id, (preds, targs) in enumerate(zip(predicitons, targets)):

                    scaled_anchors = scale_anchors(
                        self.anchors[scale_id: scale_id + 3], 
                        self.scales[scale_id],
                        self.img_width, self.img_height,
                        device=self.device
                    )

                    _, val_history = self.loss_fn(
                        preds,
                        targs,
                        scaled_anchors
                    )

                    for key in val_epoch_history.keys():
                        val_epoch_history[key] += val_history[key]

        for key in val_epoch_history.keys():
            self.val_history[key].append(
                val_epoch_history[key] / len(self.v_dataset)
            )

        print(f"EPOCH {epoch} AVG VAL LOSS {np.round(self.val_history['total_loss'][-1], 3)}")

        return None

    def fit(self, num_epochs, save_model_to: str):
        if None in [self.model, self.t_dataset, self.v_dataset, self.loss_fn,
                    self.anchors, self.scales, self.optimizer]:
            err_msg = "model, t_dataset, v_dataset, loss_fn, anchors, scales, "
            err_msg += "optimizer must not be None"
            raise Exception(err_msg)

        assert self.model is nn.Module

        best_loss = 1e6
        for i in range(1, num_epochs + 1):

            self.model.train()
            self._train_one_epoch(epoch=i)

            self.model.eval()
            self._validate_one_epoch(epoch=i)

            avg_epoch_val_loss = np.mean(self.val_history['total_loss'][-1])
            if avg_epoch_val_loss < best_loss:
                best_loss = avg_epoch_val_loss
                torch.save(self.model.state_dict(), save_model_to)

        return None
