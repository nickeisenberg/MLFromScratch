from trfc.algo.base import ObjDet
from torch.optim import Adam
from torch.cuda import is_available
from torch import save as save_model
from experiment.config.neural_network.layers import YOLOv5
from experiment.config.neural_network.loss import YOLOLoss

device = "cuda" if is_available() else "cpu"

class YOLOv5Network(ObjDet):
    def __init__(self):
        super().__init__(name="yolov5_test")
        self.yolov5 = YOLOv5(3, 10)
        self.optimizer = Adam(self.parameters())
        self.loss_fn = YOLOLoss(device)


    def forward(self, inputs):
        return self.yolov5(inputs)


    def batch_pass(self, inputs, targets):
        predicitons = self.forward(inputs)

        loss, batch_history = self.loss_fn(
            predicitons, targets
        )
        
        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, batch_history


    def evaluate(self, inputs):
        pass
        
    
    def save(self, path: str):
        save_model(self.state_dict(), path)

yolov5Network = YOLOv5Network()
