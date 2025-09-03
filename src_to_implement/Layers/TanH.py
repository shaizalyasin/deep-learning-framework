import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()       # sets trainable=False, testing_phase=False
        self.trainable = False   # explicit, since it has no weights

    def forward(self, X):
        # store output so backward can use it
        self.y = np.tanh(X)
        return self.y

    def backward(self, dY):
        # derivative of tanh is (1 - tanh^2)
        return dY * (1 - self.y**2)
