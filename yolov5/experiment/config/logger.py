from trfc.logger import CSVLogger
import numpy as np

class Logger(CSVLogger):
    def __init__(self):

        super().__init__()

        self.best_loss = 1e6

    def check(self) -> bool:
        avg_epoch_loss = np.mean(self.history["total_loss"])
    
        if avg_epoch_loss < self.best_loss:
            self.best_loss = avg_epoch_loss
            return True
        else:
            return False

logger = Logger()
