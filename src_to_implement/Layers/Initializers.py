import numpy as np

class Constant:
    def __init__(self, const_value = 0.1):
        self.const_value = const_value

    def initialize(self,  weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.const_value)

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0.0, 1.0, weights_shape)

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # Formula taken from Slide#5
        sigma = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, sigma, weights_shape)

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # Formula taken from Slide#6
        sigma = np.sqrt(2.0 / fan_in)
        return np.random.normal(0.0, sigma, weights_shape)