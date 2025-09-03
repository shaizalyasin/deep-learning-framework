# src_to_implement/Layers/Base.py

class BaseLayer:
    def __init__(self):
        # By default layers are not trainable unless they override this
        self.trainable = False
        # Flag to switch between train / test behavior (e.g. Dropout, BatchNorm)
        self.testing_phase = False

        # Optional members that some layers may use
        self.weight = None        # e.g. FullyConnected might set this
        self.input_tensor = None  # can store the last input in forward()

    def forward(self, X):
        """
        Compute the forward pass.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Layer must implement forward()")

    def backward(self, dY):
        """
        Compute the backward pass.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Layer must implement backward()")
