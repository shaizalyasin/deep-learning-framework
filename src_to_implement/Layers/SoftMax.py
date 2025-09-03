import numpy as np

from Layers.Base import BaseLayer

class SoftMax(BaseLayer): # SoftMax activation function layer
    def __init__(self):
        super().__init__()
        self.predictions = None  # stores output of forward pass

    def forward(self, input_tensor):
        shifted_tensor = input_tensor - np.max(input_tensor, axis=1, keepdims=True) # to prevent overflow in exp calculation
        exp_tensor = np.exp(shifted_tensor) # exponentiate the shifted tensor
        self.predictions = exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True) # normalize to get probabilities
        return self.predictions # this returns the softmax probabilities

    def backward(self, error_tensor): # Backward pass computes the gradient of the loss with respect to the input tensor
        # Efficient Jacobian-vector product for SoftMax
        return self.predictions * (
            error_tensor - np.sum(error_tensor * self.predictions, axis=1, keepdims=True)
        )
