import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None  # original shape for backward pass

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape  # (batch_size, channels, H, W)
        # (11, 2, 6, 9) -> (11, 2*6*9) -> (11, 108)
        batch_size = self.input_shape[0]
        # print(input_tensor.reshape(batch_size, -1))
        return input_tensor.reshape(batch_size, -1)

    def backward(self, error_tensor):
        # (11, 108) -> (11, 2, 6, 9)
        # print(error_tensor.reshape(self.input_shape))
        return error_tensor.reshape(self.input_shape)  # Restore original shape
