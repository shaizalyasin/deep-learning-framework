import numpy as np

class L1_Regularizer:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def norm(self, W: np.ndarray) -> float:
        # sum of absolute values times alpha
        return self.alpha * np.sum(np.abs(W))

    def calculate_gradient(self, W: np.ndarray) -> np.ndarray:
        # elementwise sign times alpha
        return self.alpha * np.sign(W)


class L2_Regularizer:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def norm(self, W: np.ndarray) -> float:
        # sum of squares times alpha
        return self.alpha * np.sum(W * W)

    def calculate_gradient(self, W: np.ndarray) -> np.ndarray:
        # elementwise alpha*W
        return self.alpha * W
