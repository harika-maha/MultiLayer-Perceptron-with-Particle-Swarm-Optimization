import numpy as np

class Cost:
    def __init__(self, y_pred, y):
        self.y = y
        self.y_pred = y_pred

    def get_mae(self):
        return np.mean(np.abs(self.y - self.y_pred))

    def get_mse(self):
        return np.mean((self.y - self.y_pred)**2)