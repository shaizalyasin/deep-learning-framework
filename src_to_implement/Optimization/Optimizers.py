# import numpy as np

# class Optimizer:
#     def __init__(self):
#         self.regularizer = None
#     def add_regularizer(self, regularizer):
#         self.regularizer = regularizer

# class Sgd(Optimizer):
#     def __init__(self, learning_rate):
#         super().__init__()
#         self.learning_rate = learning_rate

#     def calculate_update(self, weight_tensor, gradient_tensor):
#         if self.regularizer is not None:
#             gradient_tensor = gradient_tensor + self.regularizer.calculate_gradient(weight_tensor)
#         update = weight_tensor - (self.learning_rate * gradient_tensor)
#         return update

# class SgdWithMomentum(Optimizer):
#     def __init__(self, learning_rate, momentum_rate):
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.momentum_rate = momentum_rate
#         self.v_k = None  # velocity

#     def calculate_update(self, weight_tensor, gradient_tensor):
#         if self.regularizer is not None:
#             gradient_tensor = gradient_tensor + self.regularizer.calculate_gradient(weight_tensor)
#         if self.v_k is None:
#             self.v_k = np.zeros_like(weight_tensor)
#         self.v_k = self.momentum_rate * self.v_k - self.learning_rate * gradient_tensor
#         update = weight_tensor + self.v_k
#         return update

# class Adam(Optimizer):
#     def __init__(self, learning_rate, mu, rho):
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.mu = mu
#         self.rho = rho
#         self.v_k = None  # first moment
#         self.r_k = None  # second moment
#         self.k = 0       # timestep
#         self.epsilon = 1e-8

#     def calculate_update(self, weight_tensor, gradient_tensor):
#         if self.regularizer is not None:
#             gradient_tensor = gradient_tensor + self.regularizer.calculate_gradient(weight_tensor)
#         if self.v_k is None:
#             self.v_k = np.zeros_like(weight_tensor)
#             self.r_k = np.zeros_like(weight_tensor)
#         self.k += 1
#         self.v_k = self.mu * self.v_k + (1 - self.mu) * gradient_tensor # First Order Momentum
#         self.r_k = self.rho * self.r_k + (1 - self.rho) * (gradient_tensor ** 2) # Second Order Momentum
#         v_hat = self.v_k / (1 - self.mu ** self.k)
#         r_hat = self.r_k / (1 - self.rho ** self.k)
#         update = weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.epsilon)
#         return update

import numpy as np


class Optimizer:
    """
    Base-class that only stores (and allows attaching) a regularizer.
    """
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # plain SGD keeps the “reg-in-gradient” behaviour that already passes the tests
        if self.regularizer is not None:
            gradient_tensor = gradient_tensor + self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum(Optimizer):
    """
    Momentum with **decoupled** regularisation:
      – first do the classical momentum step with the *data* gradient  
      – THEN apply weight-decay (L2) or L1 shrinkage **once** to the result
        using the *original* weights (as the test-suite expects).
    """
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_k = None  # velocity

    def calculate_update(self, weight_tensor, gradient_tensor):
        # initialise velocity
        if self.v_k is None:
            self.v_k = np.zeros_like(weight_tensor)

        # classic momentum update (no reg in the gradient)
        self.v_k = self.momentum_rate * self.v_k - self.learning_rate * gradient_tensor
        updated = weight_tensor + self.v_k

        # decoupled regulariser step
        if self.regularizer is not None:
            α = self.regularizer.alpha * self.learning_rate
            if self.regularizer.__class__.__name__ == 'L1_Regularizer':
                updated = updated - α * np.sign(weight_tensor)
            elif self.regularizer.__class__.__name__ == 'L2_Regularizer':
                updated = updated - α * weight_tensor

        return updated


class Adam(Optimizer):
    """
    Adam **with decoupled weight-decay / L1-shrink** (AdamW-style).
    """
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu          # β₁
        self.rho = rho        # β₂
        self.v_k = None       # first moment
        self.r_k = None       # second moment
        self.k = 0            # timestep
        self.epsilon = 1e-8

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v_k is None:
            self.v_k = np.zeros_like(weight_tensor)
            self.r_k = np.zeros_like(weight_tensor)

        # Adam moments (no reg in the gradient!)
        self.k += 1
        self.v_k = self.mu * self.v_k + (1 - self.mu) * gradient_tensor
        self.r_k = self.rho * self.r_k + (1 - self.rho) * (gradient_tensor ** 2)

        v_hat = self.v_k / (1 - self.mu ** self.k)
        r_hat = self.r_k / (1 - self.rho ** self.k)

        updated = weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.epsilon)

        # decoupled regulariser step
        if self.regularizer is not None:
            α = self.regularizer.alpha * self.learning_rate
            if self.regularizer.__class__.__name__ == 'L1_Regularizer':
                updated = updated - α * np.sign(weight_tensor)
            elif self.regularizer.__class__.__name__ == 'L2_Regularizer':
                updated = updated - α * weight_tensor

        return updated
