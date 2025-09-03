import numpy as np
import copy
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=0.0, high=1.0, size=(input_size + 1, output_size))
        # self.weights[-1, :] = np.random.uniform(low=0.0, high=1.0, size=(1, output_size))

        self.input_tensor_augmented = None
        self._optimizer = None
        self._gradient_weights = None

    def forward(self, input_tensor):
        bias_column = np.ones((input_tensor.shape[0], 1))
        self.input_tensor_augmented = np.concatenate((input_tensor, bias_column), axis=1)
        return np.dot(self.input_tensor_augmented, self.weights)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        # self._optimizer = copy.deepcopy(optimizer)
        self._optimizer = optimizer

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(self.input_tensor_augmented.T, error_tensor)

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        # Gradient for the previous layer's input. Exclude the bias part of weights.
        return np.dot(error_tensor, self.weights[:-1, :].T)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    def zero_grad(self):
        self._gradient_weights = np.zeros_like(self.weights)

    def initialize(self, weights_initializer, bias_initializer):
        # weight part initialization
        self.weights[:-1, :] = weights_initializer.initialize((self.input_size, self.output_size), self.input_size,
                                                              self.output_size)
        # bias part initialization
        self.weights[-1, :] = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
