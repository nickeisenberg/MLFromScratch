import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import numpy as np
from torch.utils.data.dataset import random_split
from model.loss import YoloV3Loss
from utils import scale_anchors

class Model:
    def __init__(
        self, 
        model: nn.Module,
        save_model_to: str,
        loss_fn: YoloV3Loss, 
        optimizer: Optimizer, 
        t_dataset: Dataset,
        v_dataset: Dataset,
        batch_size: int,
        device: str,
        scales: torch.Tensor,
        anchors: torch.Tensor,
        notify_after: int
        ):
        self.model = model
        self.save_model_to = save_model_to
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.t_dataset = t_dataset
        self.v_dataset = v_dataset
        self.t_dataloader = DataLoader(self.t_dataset, batch_size, shuffle=True)
        self.v_dataloader = DataLoader(self.v_dataset, batch_size, shuffle=False)
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
        self.scales = scales.to(device)
        self.anchors = anchors.to(device)
        self.notify_after = notify_after
        _, self.img_height, self.img_width = t_dataset.__getitem__(0)[0].shape

    def train_one_epoch(self, epoch):

        for key in self.history.keys():
            self.history[key][epoch] = []

        best_loss = 1e6
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

    def validate_one_epoch(self, epoch):
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

    def fit(self, num_epochs):
        best_loss = 1e6
        for i in range(1, num_epochs + 1):

            self.model.train()
            self.train_one_epoch(epoch=i)

            self.model.eval()
            self.validate_one_epoch(epoch=i)

            avg_epoch_val_loss = np.mean(self.val_history['total_loss'][-1])
            if avg_epoch_val_loss < best_loss:
                best_loss = avg_epoch_val_loss
                torch.save(self.model.state_dict(), self.save_model_to)

        return None
