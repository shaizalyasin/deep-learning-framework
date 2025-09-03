import numpy as np

from Layers.Base import BaseLayer


class ReLU(BaseLayer): # Rectified Linear Unit (ReLU) activation function layer
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor): # Forward pass applies ReLU activation
        output_tensor = np.maximum(0, input_tensor) # ReLU activation function
        self.input_tensor = input_tensor # store input tensor for backward pass
        return output_tensor

    def backward(self, error_tensor): #backward pass computes gradient of ReLU
        return error_tensor * (self.input_tensor > 0) #this returns the gradient of the ReLU function,
        # which is 1 for positive inputs and 0 for non-positive inputs
