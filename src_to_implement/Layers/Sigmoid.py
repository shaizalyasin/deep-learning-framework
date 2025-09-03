import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, X):
        self.y = 1.0 / (1.0 + np.exp(-X))
        return self.y

    def backward(self, dY):
        # derivative of σ is σ*(1−σ)
        return dY * (self.y * (1 - self.y))
