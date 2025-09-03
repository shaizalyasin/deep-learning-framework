import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.predictions = None # stores output of forward pass

    def forward(self, prediction_tensor, label_tensor):
        shifted_tensor = np.log(prediction_tensor + np.finfo('float').eps) # to avoid log(0)
        self.predictions = prediction_tensor # store predictions for backward pass
        return np.sum(-shifted_tensor * label_tensor) # compute loss

    def backward(self, label_tensor): # compute gradient
        return -(np.divide(label_tensor, self.predictions))  # this is the gradient of the loss with respect to the predictions