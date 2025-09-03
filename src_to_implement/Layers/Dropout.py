# src_to_implement/Layers/Dropout.py

import numpy as np
from .Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, p_keep: float):
        """
        p_keep: probability of keeping each activation during training.
        """
        super().__init__()        # sets trainable=False, testing_phase=False
        self.p = p_keep           # keep probability
        self.trainable = False    # no weights to learn

    def forward(self, X):
        """
        During training (testing_phase=False), zero out units with prob (1-p)
        and scale up the rest by 1/p (inverted dropout).
        During testing, pass X through unchanged.
        """
        if not self.testing_phase:
            # mask of 0/1 scaled by 1/p
            self.mask = (np.random.rand(*X.shape) < self.p) / self.p
            return X * self.mask
        else:
            self.mask = None  # clear mask in test mode
            return X

    def backward(self, dY):
        """
        Backprop only through the kept units.
        """
        if not self.testing_phase:
            return dY * self.mask
        else:
            return dY
