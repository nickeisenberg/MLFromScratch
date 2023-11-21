import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import numpy as np
from model.loss import YoloV3Loss

class Model:
    def __init__(
        self, 
        model: nn.Module, 
        loss: YoloV3Loss, 
        optimizer: Optimizer, 
        t_dataset: Dataset,
        v_dataset: Dataset,
        batch_size: int
        ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.t_dataset = t_dataset
        self.t_dataloader = DataLoader(t_dataset, batch_size, shuffle=True)
        self.v_dataset = v_dataset

    def train_one_epoch(self, epoch):
    
        epoch_loss = 0

        for batch_num, (images, targets) in enumerate(self.t_dataloader):

            if batch_num + 1 % 100 == 0:
                batch_loss = np.round(
                    self.loss.t_history['total_loss'][-1], 
                    3
                )
                print(f"BATCH {batch_num} LOSS {batch_loss}")

            self.optimizer.zero_grad()

            prediciton = self.model(images)

            loss = self.loss(prediciton[0], targets[0])
            loss += self.loss(prediciton[1], targets[1])
            loss += self.loss(prediciton[2], targets[2])

            epoch_loss += loss.item()

            loss.backward()

            self.optimizer.step()

        print(f"EPOCH {epoch} AVG LOSS {np.round(epoch_loss / len(self.t_dataset), 3)}")

    def validate_one_epoch(self):
        
        running_loss = 0
        for images, targets in self.v_dataset):

            with torch.no_grad():
                prediciton = self.model(images)
                loss = self.loss(prediciton[0], targets[0], validate=True)
                loss += self.loss(prediciton[1], targets[1], validate=True)
                loss += self.loss(prediciton[2], targets[2], validate=True)

        avg_loss = np.round(running_loss / len(self.v_dataset), 3)

        print(f"EPOCH {epoch}  LOSS {avg_loss}")

        return None







