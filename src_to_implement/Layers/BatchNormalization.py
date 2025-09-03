# src_to_implement/Layers/BatchNormalization.py

import numpy as np
from .Base import BaseLayer

class BatchNormalization(BaseLayer):
    def __init__(self, num_features, momentum=0.9, eps=1e-10):
        super().__init__()
        self.trainable = True
        self.testing_phase = False
        self.momentum = momentum
        self.eps = eps
        self.num_features = num_features
        # gamma (scale) and beta (shift)
        self.weights = np.ones(num_features)  # gamma
        self.bias = np.zeros(num_features)    # beta
        self.gradient_weights = np.zeros(num_features)
        self.gradient_bias = np.zeros(num_features)
        # running mean/var for test phase
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        # cache for backward
        self.cache = None
        self.last_input_shape = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.num_features,), self.num_features, 1)
        self.bias = bias_initializer.initialize((self.num_features,), self.num_features, 1)

    def forward(self, X):
        is_conv = (X.ndim == 4)
        if is_conv:
            self.last_input_shape = X.shape
            X_flat = self.reformat(X)
        else:
            self.last_input_shape = X.shape
            X_flat = X
        if not self.testing_phase:
            batch_mean = np.mean(X_flat, axis=0)
            batch_var = np.var(X_flat, axis=0)
            self.batch_mean = batch_mean
            self.batch_var = batch_var
            # Initialize running stats to first batch's stats if first pass
            if not hasattr(self, '_running_initialized') or not self._running_initialized:
                self.running_mean = batch_mean.copy()
                self.running_var = batch_var.copy()
                self._running_initialized = True
            else:
                self.running_mean = (self.momentum * self.running_mean + (1 - self.momentum) * batch_mean).copy()
                self.running_var = (self.momentum * self.running_var + (1 - self.momentum) * batch_var).copy()
            self.X_centered = X_flat - batch_mean
            self.std_inv = 1. / np.sqrt(batch_var + self.eps)
            self.X_norm = self.X_centered * self.std_inv
            out = self.weights * self.X_norm + self.bias
        else:
            X_centered = X_flat - self.running_mean
            X_norm = X_centered / np.sqrt(self.running_var + self.eps)
            out = self.weights * X_norm + self.bias
        if is_conv:
            out = self.reformat(out)
        return out

    def backward(self, dY):
        is_conv = (dY.ndim == 4)
        if is_conv:
            self.last_input_shape = dY.shape
            dY_flat = self.reformat(dY)
        else:
            dY_flat = dY
        N = dY_flat.shape[0]
        # Gradients for gamma and beta
        self.gradient_weights = np.sum(dY_flat * self.X_norm, axis=0)
        self.gradient_bias = np.sum(dY_flat, axis=0)
        # Gradient for input
        dX_norm = dY_flat * self.weights
        dvar = np.sum(dX_norm * self.X_centered * -0.5 * self.std_inv**3, axis=0)
        dmean = np.sum(-dX_norm * self.std_inv, axis=0) + dvar * np.mean(-2. * self.X_centered, axis=0)
        dX = dX_norm * self.std_inv + dvar * 2 * self.X_centered / N + dmean / N
        if is_conv:
            dX = self.reformat(dX)
        # Update weights and bias if optimizer is set
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
        return dX

    def reformat(self, X):
        # If input is 4D (N, C, H, W), reshape to (N*H*W, C)
        if X.ndim == 4:
            N, C, H, W = X.shape
            return X.transpose(0, 2, 3, 1).reshape(-1, C)
        # If input is 2D (N*H*W, C), reshape back to (N, C, H, W)
        elif X.ndim == 2:
            if hasattr(self, 'last_input_shape') and self.last_input_shape is not None:
                N, C, H, W = self.last_input_shape
                return X.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            else:
                return X
        else:
            return X
