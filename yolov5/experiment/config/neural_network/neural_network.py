import torch
from trfc.algo.base import ObjDet
from torch.optim import Adam
from torch.cuda import is_available
from torch import save as save_model
from experiment.config.neural_network.layers import YOLOv5
from experiment.config.neural_network.loss import YOLOLoss
from experiment.config.dataset import anchors, scales

device = "cuda" if is_available() else "cpu"

device = "cpu"

class YOLOv5Network(ObjDet):
    def __init__(self):
        super().__init__(name="yolov5_test")
        self.yolov5 = YOLOv5(1, 5)
        self.optimizer = Adam(self.parameters())
        self.loss_fn = YOLOLoss(device)

        self.scales = scales.to(device)
        self.anchors = anchors.to(device)
        self.scaled_anchors = [
            scale_anchors(
                self.anchors[scale_id * 3: (scale_id + 1) * 3], 
                self.scales[scale_id],
                self.img_width, self.img_height,
                device=self.device
            )
        for scale_id in range(len(self.scales))]

    def forward(self, inputs):
        return self.yolov5(inputs)


    def batch_pass(self, inputs, targets):
        with torch.cuda.amp.autocast_mode.autocast():

            predicitons = self.model(inputs)
            
            batch_loss = torch.zeros(1, requires_grad=True).to(self.device)
            for scale_id, (preds, targs) in enumerate(zip(predicitons, targets)):

                _batch_loss, batch_history = self.loss_fn(
                    preds,
                    targs,
                    self.scaled_anchors[scale_id]
                )

                batch_loss += _batch_loss

            if self.training:
                self.optimizer.zero_grad()
                self.scaler.scale(batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        return loss, batch_history


    def evaluate(self, inputs):
        pass
        
    
    def save(self, path: str):
        save_model(self.state_dict(), path)

yolov5Network = YOLOv5Network()
