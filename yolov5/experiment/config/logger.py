from trfc.logger import CSVLogger
import numpy as np

class Logger(CSVLogger):
    def __init__(self):

        super().__init__()

        self.train_best_loss = 1e6
        self.val_best_loss = 1e6

    def check(self, train) -> bool:
        avg_epoch_loss = np.mean(self.history["total_loss"])
        
        if train:
            if avg_epoch_loss < self.train_best_loss:
                self.train_best_loss = avg_epoch_loss
                return True
            else:
                return False
        else:
            if avg_epoch_loss < self.val_best_loss:
                self.val_best_loss = avg_epoch_loss
                return True
            else:
                return False

logger = Logger()
