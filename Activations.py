import numpy as np

class Activations:

    def tanh(self, Z):
        return np.tanh(Z)

    def relu(self, Z):
        return np.maximum(0, Z)

    def logistic(self, Z):
        Z_safe = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z_safe))
