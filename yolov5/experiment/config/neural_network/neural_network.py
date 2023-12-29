import torch
from trfc.algo.base import ObjDet
from trfc.dataset.objdet.utils.yolo import scale_anchors
from torch.optim import Adam
from torch.cuda import is_available
from torch import save as save_model
from experiment.config.neural_network.layers import YOLOv5
from experiment.config.neural_network.loss import YOLOLoss
from experiment.config.dataset import anchors, scales

class YOLOv5Network(ObjDet):
    def __init__(self, device):
        super().__init__(name="yolov5_test")
        
        self.device = device

        self.yolov5 = YOLOv5(1, 5).to(device)
        self.optimizer = Adam(self.parameters(), lr=.0001, weight_decay=.00001)
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss_fn = YOLOLoss(device)

        self.scales = scales.to(device)
        self.anchors = anchors.to(device)
        self.scaled_anchors = [
            scale_anchors(
                self.anchors[scale_id * 3: (scale_id + 1) * 3], 
                self.scales[scale_id],
                640, 512,
                device=device
            )
        for scale_id in range(len(self.scales))]

    def forward(self, inputs):
        return self.yolov5(inputs)

    def batch_pass(self, inputs, targets):
        batch_history = {
            "box_loss": [],
            "object_loss": [],
            "no_object_loss": [],
            "class_loss": [],
            "total_loss": [],
        }


        inputs = inputs.to(self.device, torch.float32)
        with torch.cuda.amp.autocast():

            predicitons = self.forward(inputs)
            targets = tuple([target.to(self.device, torch.float32) for target in targets])
            
            batch_loss = torch.zeros(
                1, requires_grad=True).to(self.device, torch.float32)
            for scale_id, (preds, targs) in enumerate(zip(predicitons, targets)):
                _batch_loss, _batch_history = self.loss_fn(
                    preds,
                    targs,
                    self.scaled_anchors[scale_id]
                )

                batch_loss = batch_loss + _batch_loss.to(torch.float32)
                
                for key in batch_history.keys():
                    batch_history[key].append(_batch_history[key])

            if self.training:
                self.optimizer.zero_grad()
                self.scaler.scale(batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        return batch_loss, batch_history

    def evaluate(self, inputs):
        pass
        
    def save(self, path: str):
        save_model(self.state_dict(), path)

device = "cuda" if is_available() else "cpu"

yolov5Network = YOLOv5Network(device=device)

# for inps, targs in train_dataloader:
#     inps, targs = inps.to(device), [targ.to(device) for targ in targs]
#     x = yolov5Network.batch_pass(inps, targs)
#     break
