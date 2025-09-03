import numpy as np
import copy
from scipy.signal import correlate2d, convolve2d
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape, stride_shape)
        elif len(stride_shape) == 1:
            self.stride_shape = (stride_shape[0], stride_shape[0])
        else:
            self.stride_shape = stride_shape

        # if it's a 1D convolution (treated as 2D with height/width of 1 for kernel)
        self.is_1d = len(convolution_shape) == 2
        if self.is_1d:
            self.input_channels, kernel_size = convolution_shape
            self.kernel_shape = (kernel_size, 1) # Kernel height, kernel width
            # Weights shape: (num_kernels, input_channels, kernel_height, kernel_width)
            self.weights = np.random.rand(num_kernels, self.input_channels, kernel_size, 1)
        else:
            self.input_channels, kh, kw = convolution_shape
            self.kernel_shape = (kh, kw) # Kernel height, kernel width
            self.weights = np.random.rand(num_kernels, self.input_channels, kh, kw)

        self.bias = np.random.rand(num_kernels)

        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer_weights = None
        self._optimizer_bias = None
        self.input_tensor = None # Stores the original input tensor for backward pass
        self.padded_input_fwd = None # Stores the padded input from forward pass for backward use

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, optimizer):
        # Deep copy ensures separate optimizer instances for weights and bias
        self._optimizer_weights = copy.deepcopy(optimizer)
        self._optimizer_bias = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        # Fan-in and Fan-out for weight initialization
        fan_in = np.prod(self.weights.shape[1:]) # Number of connections entering a neuron
        fan_out = self.weights.shape[0] * np.prod(self.weights.shape[2:]) # Number of connections leaving a neuron
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.weights.shape[0])

    def forward(self, input_tensor):
        self.input_tensor = input_tensor # Store original input for backward pass

        # For 1D convolution add a new axis to treat it as a 2D convolution
        if self.is_1d:
            input_tensor = input_tensor[:, :, :, np.newaxis]

        b, c, h, w = input_tensor.shape # Batch, Channels, Height, Width
        k_h, k_w = self.kernel_shape    # Kernel Height, Kernel Width
        s_h, s_w = self.stride_shape    # Stride Height, Stride Width

        pad_h_total = k_h - 1
        pad_w_total = k_w - 1

        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left

        # padding to the input tensor
        padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                              mode='constant')
        self.padded_input_fwd = padded_input # Store this padded input for the backward pass
        # Store padding amounts for backward pass
        self.forward_pad_top = pad_top
        self.forward_pad_left = pad_left

        out_h = int(np.ceil((padded_input.shape[2] - k_h + 1) / s_h))
        out_w = int(np.ceil((padded_input.shape[3] - k_w + 1) / s_w))

        # output tensor initialization for the convolution operation
        output = np.zeros((b, self.weights.shape[0], out_h, out_w))

        # Iterate over batches
        for i in range(b):
            # Iterate over each output kernel
            for k in range(self.weights.shape[0]):
                # Iterate over each input channel
                for ch in range(c):
                    # correlation between a slice of the padded input and a kernel
                    # valid mode means computation only where the kernel fully overlaps
                    conv = correlate2d(padded_input[i, ch], self.weights[k, ch], mode='valid')
                    output[i, k] += conv[::s_h, ::s_w]
                # corresponding bias after summing across all input channels for the current output kernel
                output[i, k] += self.bias[k]

        # last dimension for 1D convolution results to match expected output shape
        return output if not self.is_1d else output[:, :, :, 0]

    def backward(self, error_tensor):
        # Reshape input and error tensors to 4D for 1D convolution cases
        if self.is_1d:
            error_tensor = error_tensor[:, :, :, np.newaxis]
            input_tensor_original_shape = self.input_tensor[:, :, :, np.newaxis]
        else:
            input_tensor_original_shape = self.input_tensor

        b, c, h, w = input_tensor_original_shape.shape # Batch, Channels, Original Height, Original Width
        k_out, k_in, k_h, k_w = self.weights.shape     # Output Kernels, Input Kernels, Kernel Height, Kernel Width
        s_h, s_w = self.stride_shape                    # Stride Height, Stride Width

        out_h = error_tensor.shape[2] # Height of the error tensor
        out_w = error_tensor.shape[3] # Width of the error tensor

        # gradients initialization for weights and bias
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        # gradient initialization with respect to the input tensor
        grad_input_unpadded = np.zeros_like(input_tensor_original_shape)

        pad_v_total_grad_w = k_h - 1
        pad_h_total_grad_w = k_w - 1
        pad_v_before_grad_w = pad_v_total_grad_w // 2
        pad_v_after_grad_w = pad_v_total_grad_w - pad_v_before_grad_w
        pad_h_before_grad_w = pad_h_total_grad_w // 2
        pad_h_after_grad_w = pad_h_total_grad_w - pad_h_before_grad_w

        # Iterate over batches
        for i in range(b):
            # Iterate over each output kernel
            for k in range(k_out):
                # Gradient with respect to bias
                self._gradient_bias[k] += np.sum(error_tensor[i, k])

                # This tensor will have the spatial dimensions
                # of the input (h, w) with error values placed at strided locations.
                upsampled_error = np.zeros((h, w))
                for y in range(out_h):
                    for x in range(out_w):
                        upsampled_error[y * s_h, x * s_w] = error_tensor[i, k, y, x]

                # Iterate over each input channel
                for ch in range(c):
                    # padding ensures that `correlate2d` will output the correct kernel shape.
                    input_for_weight_grad_channel = np.pad(input_tensor_original_shape[i, ch],
                                                           ((pad_v_before_grad_w, pad_v_after_grad_w),
                                                            (pad_h_before_grad_w, pad_h_after_grad_w)),
                                                           mode='constant')

                    # Gradient with respect to weights:
                    self._gradient_weights[k, ch] += correlate2d(input_for_weight_grad_channel, upsampled_error,
                                                                 mode='valid')
                    # Gradient with respect to input:
                    conv_full_result = convolve2d(upsampled_error, self.weights[k, ch], mode='full')

                    grad_input_slice = conv_full_result[
                        self.forward_pad_top : self.forward_pad_top + h,
                        self.forward_pad_left : self.forward_pad_left + w
                    ]
                    grad_input_unpadded[i, ch] += grad_input_slice

        if self._optimizer_weights:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        # last dimension for 1D convolution results to match expected output shape
        return grad_input_unpadded if not self.is_1d else grad_input_unpadded[:, :, :, 0]

# Forward Pass
# x = np.random.randn(2, 3, 32, 32)  # batch=2, channels=3, 32x32 image
# conv = Conv((1,1), (3, 3, 3), 5)   # 5 filters, 3x3, 3 input channels
# out = conv.forward(x)
# print(out.shape)  # (2, 5, 32, 32)

# Backward Pass
# x = np.random.randn(2, 3, 32, 32)
# conv = Conv((1, 1), (3, 3, 3), 5)
# out = conv.forward(x)
# err = np.random.randn(*out.shape)
# back = conv.backward(err)
# print(back.shape)  # (2, 3, 32, 32)

