import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        # similar to Conv layer
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape, stride_shape)
        elif len(stride_shape) == 1:
            self.stride_shape = (stride_shape[0], stride_shape[0])
        else:
            self.stride_shape = stride_shape

        self.pooling_shape = pooling_shape  # Store pooling window dimensions (p_h, p_w)

        # Variables to store information for the backward pass
        self.input_shape = None
        # self.max_indices will store the (batch, channel, h, w) index of the max value
        self.max_indices = []

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        b, c, h, w = input_tensor.shape
        p_h, p_w = self.pooling_shape
        s_h, s_w = self.stride_shape

        out_h = (h - p_h) // s_h + 1
        out_w = (w - p_w) // s_w + 1

        # Initialize the output tensor with zeros
        output = np.zeros((b, c, out_h, out_w))

        # max_indices reset for the current forward pass
        self.max_indices = []

        # Iterate over batch, channels, and the output spatial dimensions
        for batch in range(b):
            for channel in range(c):
                for out_y in range(out_h):
                    for out_x in range(out_w):
                        h_start = out_y * s_h
                        h_end = h_start + p_h
                        w_start = out_x * s_w
                        w_end = w_start + p_w

                        window = input_tensor[batch, channel, h_start:h_end, w_start:w_end]

                        # the maximum value in the current window
                        max_val = np.max(window)
                        output[batch, channel, out_y, out_x] = max_val

                        # Store the local index of the maximum value.
                        local_max_pos = np.unravel_index(np.argmax(window), window.shape)

                        # Store the global index of the maximum value.
                        # tells us exactly where the gradient should be propagated back.
                        global_max_h = h_start + local_max_pos[0]
                        global_max_w = w_start + local_max_pos[1]
                        self.max_indices.append((batch, channel, global_max_h, global_max_w))

        return output

    def backward(self, error_tensor):
        # Initialize the gradient tensor for the input with zeros
        grad_input = np.zeros(self.input_shape)

        b, c, out_h, out_w = error_tensor.shape
        # p_h, p_w = self.pooling_shape
        # s_h, s_w = self.stride_shape
        idx_counter = 0

        # Iterate over batch, channels, and the output
        for batch in range(b):
            for channel in range(c):
                for out_y in range(out_h):
                    for out_x in range(out_w):
                        # incoming error value for the current output element
                        error_val = error_tensor[batch, channel, out_y, out_x]

                        # Retrieve the stored global index (batch, channel, global_h, global_w)
                        # that corresponds to this output element.
                        # We must ensure that the order of appending to self.max_indices
                        # in forward matches the order of iterating through error_tensor here.
                        max_b, max_c, max_h, max_w = self.max_indices[idx_counter]
                        # Propagate the error value only to the position where the maximum was found
                        grad_input[max_b, max_c, max_h, max_w] += error_val

                        # next stored maximum index
                        idx_counter += 1

        return grad_input