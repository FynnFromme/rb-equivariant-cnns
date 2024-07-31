"""
Harmonic Network Layers
"""

from typing import Callable

import tensorflow as tf
from tensorflow import keras

from .ops import h_filters, h_ops


class HRangeConv(keras.Layer):
    def __init__(self, channels: int, ksize: int, strides: tuple = (1,1), padding: str = 'VALID', phase: bool = True, 
                 in_range: tuple = (0,1), out_range: tuple = (0,1), profile_initializer: keras.Initializer = None, 
                 phase_initializer: keras.Initializer = None, profile_regularizer: keras.Regularizer = None, 
                 n_rings: int | None = None, conjugate_weights: bool = True, name: str = 'HConv'):
        """The Harmonic Convolutional Layer convolves the input with grid sampled circular harmonics.
        
        In contrast to `HConv`, the rotation orders don't start at 0 and can be in any range.
        
        The input must have shape [batch_size, h, w, in_orders, complex, in_channels] and outputs a tensor of the same
        format: [batch_size, h', w', out_orders, complex, channels].

        Args:
            channels (int): The number of output channels.
            ksize (int): The size of the filter.
            strides (tuple, optional): Stride used in the conv operation. Defaults to (1,1).
            padding (str, optional): Padding of the conv operation ('VALID' or 'SAME'). Defaults to 'VALID'.
            phase (bool, optional): Whether to use a per-channel phase offset or no offset at all. Defaults to True.
            in_range (tuple, optional): The range of rotation orders of the input feature maps as a 2 tuple. Including both
                start and end. Defaults to (0,1).
            out_range (tuple, optional): The range of rotation orders of the output feature maps as a 2 tuple. Including both
                start and end. Defaults to (0,1).
            profile_initializer (Initializer, optional): Initializer used to initialize the radial profile. Defaults to None.
            phase_initializer (Initializer, optional): Initializer used to initialize the phase offset terms. Defaults to None.
            profile_regularizer (Regularizer, optional): The regularzation applied to the radial profile. Defaults to None.
            n_rings (int, optional): The number of magnitudes in the circular harmonics with distinct R values. 
                Defaults to one ring per filter unit and at least 2. Defaults to None.
            conjugate_weights (bool, optional): Whether to use conjugate weights for negative orders. Defaults to True.
            name (str, optional): The name of the layer. Defaults to 'HConv'.
        """
        super().__init__()
        self.channels = channels
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.phase = phase
        self.in_range = in_range
        self.out_range = out_range
        self.profile_initializer = profile_initializer
        self.phase_initializer = phase_initializer
        self.profile_regularizer = profile_regularizer
        self.n_rings = n_rings
        self.conjugate_weights = conjugate_weights
        self.name = name

    def build(self, input_shape: list):
        """Initializes the filter weights.

        Args:
            input_shape (list): The shape of the input.
        """
        input_channels = input_shape[5]
        shape = [self.ksize, self.ksize, input_channels, self.channels]
        
        # sepearate learnable R(r) for each rotation order x inputchannel x outputchannel
        self.q = h_filters.get_weights_dict(self, shape, self.out_range, self.in_range, 
                                                   conjugate_weights=self.conjugate_weights,
                                                   initializer=self.profile_initializer, 
                                                   regularizer=self.profile_regularizer,
                                                   n_rings=self.n_rings, name='W'+self.name)
        
        if self.phase == True:
            # seperate phase offsets for each rotation order x inputchannel x outputchannel
            self.p = h_filters.get_phase_dict(self, input_channels, self.channels, self.out_range, self.in_range, 
                                                     conjugate_weights=self.conjugate_weights, 
                                                     initializer=self.phase_initializer,
                                                     name='phase'+self.name)
        else:
            self.p = None

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs the Harmonic Convolution on the input.

        Args:
            inputs (tf.Tensor): The input tensor of shape [batch_size, h, w, in_orders, complex, in_channels].

        Returns:
            tf.Tensor: The resulting tensor of shape [batch_size, h', w', out_orders, complex, channels].
        """
        with tf.name_scope(self.name) as scope:
            # grid sampled version of filter
            # cannot be done in build(), since any manipulation of weights is not tracked there
            # thus the gradient becomes 0
            self.filters = h_filters.get_filters(self.q, filter_size=self.ksize, p=self.p, n_rings=self.n_rings)
            
            return h_ops.h_range_conv(inputs, self.filters, strides=self.strides, padding=self.padding, 
                                    out_range=self.out_range, in_range=self.in_range, 
                                    conjugate_weights=self.conjugate_weights, name=self.name)
            
    def get_grid_filters(self) -> dict[int, tuple[tf.Variable, tf.Variable]]:
        """Computes the grid sampled circular harmonic filters based on the learned weights.

        Returns:
            dict[int, tuple[tf.Variable, tf.Variable]]: Dictionary containing the real and imaginary parts of the grid 
                filters for the different rotation orders. Real and imaginary parts have both the shape 
                [filter_size, filter_size, in_channels, filters].
        """
        return h_filters.get_filters(self.q, filter_size=self.ksize, p=self.p, n_rings=self.n_rings)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'ksize': self.ksize,
            'strides': self.strides,
            'padding': self.padding,
            'phase': self.phase,
            'in_range': self.in_range,
            'out_range': self.out_range,
            'profile_initializer': self.profile_initializer,
            'phase_initializer': self.phase_initializer,
            'profile_regularizer': self.profile_regularizer,
            'n_rings': self.n_rings,
            'conjugate_weights': self.conjugate_weights,
            'name': self.name
        })
        return config


class HConv(HRangeConv):
    def __init__(self, channels: int, ksize: int, strides: tuple = (1,1), padding: str = 'VALID', phase: bool = True, 
                 max_order: int = 1, profile_initializer: keras.Initializer = None, 
                 phase_initializer: keras.Initializer = None, profile_regularizer: keras.Regularizer = None, 
                 n_rings: int | None = None, conjugate_weights: bool = True, name: str = 'HConv'):
        """The Harmonic Convolutional Layer convolves the input with grid sampled circular harmonics.
        
        The input must have shape [batch_size, h, w, in_orders, complex, in_channels] and outputs a tensor of the same
        format: [batch_size, h', w', out_orders, complex, channels].

        Args:
            channels (int): The number of output channels.
            ksize (int): The size of the filter.
            strides (tuple, optional): Stride used in the conv operation. Defaults to (1,1).
            padding (str, optional): Padding of the conv operation ('VALID' or 'SAME'). Defaults to 'VALID'.
            phase (bool, optional): Whether to use a per-channel phase offset or no offset at all. Defaults to True.
            max_order (int, optional): The maximum rotation order, e.g. max_order=2 uses 0,1,2. Defaults to 1.
            profile_initializer (Initializer, optional): Initializer used to initialize the radial profile. Defaults to None.
            phase_initializer (Initializer, optional): Initializer used to initialize the phase offset terms. Defaults to None.
            profile_regularizer (Regularizer, optional): The regularzation applied to the radial profile. Defaults to None.
            n_rings (int, optional): The number of magnitudes in the circular harmonics with distinct R values. 
                Defaults to one ring per filter unit and at least 2. Defaults to None.
            conjugate_weights (bool, optional): Whether to use conjugate weights for negative orders. Defaults to True.
            name (str, optional): The name of the layer. Defaults to 'HConv'.
        """
        in_range = None
        out_range = (0, max_order)
        super().__init__(channels, ksize, strides, padding, phase, in_range, out_range, profile_initializer, phase_initializer,
                         profile_regularizer, n_rings, conjugate_weights, name)
        
    def build(self, input_shape: list):
        """Initializes the filter weights.

        Args:
            input_shape (list): The shape of the input.
        """
        in_max_order = input_shape[3]-1
        self.in_range = (0, in_max_order)
        
        return super().build(input_shape)


class HBatchNorm(keras.layers.BatchNormalization):
    def __init__(self, nonlin: Callable = keras.activations.relu, momentum: float = 0.99, epsilon: float = 0.001, 
                 magnitude_eps: float = 1e-4, name: str = 'HBatchNorm', **kwargs):
        """The Harmonic Batch Normalization Layer applies batch normalization as well as a pointwise nonlinearty
        to the magnitudes of the input.
        
        The input must have shape [batch_size, h, w, in_orders, complex, in_channels]. The output has the same shape.

        Args:
            fn (Callable, optional): The nonlinearity applied to the result. MUST map to non-negative reals R+.
                Defaults to keras.activations.relu.
            momentum (float, optional): Momentum for the moving average. Defaults to 0.99.
            epsilon (float, optional): Small float added to the variance to avoid dividing by zero. Defaults to 0.001.
            magnitude_eps (float, optional): Regularization since gradient of magnitudes is infinite at zero. Defaults to 1e-4.
            name (str, optional): The name of the layer. Defaults to 'HBatchNorm'.
        """
        super().__init__(axis=-1, momentum=momentum, epsilon=epsilon, name=name, **kwargs)
        self.nonlin = nonlin
        self.magnitude_eps = magnitude_eps
        
    def build(self, input_shape: list):
        """Initializes the beta and gamma parameters of the layer.

        Args:
            input_shape (list): The shape of the input.
        """
        magnitude_shape = list(input_shape)
        magnitude_shape[4] = 1
        super().build(magnitude_shape)
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Applies batch normalization and a nonlinearity to the magnitudes of the input.

        Args:
            inputs (tf.Tensor): The input tensor.
            training (bool, optional): Whether in training or evaluation mode. Defaults to False.

        Returns:
            tf.Tensor: The output tensor of the same shape as the input.
        """
        with tf.name_scope(self.name) as scope:
            magnitudes = h_ops.stack_magnitudes(inputs, self.magnitude_eps, keepdims=True)
        
            # perform batch normalization on the magnitudes
            normalized_magnitudes = super().call(magnitudes, training=training)
            normalized_magnitudes = self.nonlin(normalized_magnitudes)
            
            # apply nonlinearity on the normalized magnitudes and update the magnitudes of x
            return h_ops.update_magnitudes(inputs, normalized_magnitudes, magnitudes)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'nonlin': self.nonlin,
            'magnitude_eps': self.magnitude_eps
        })
        return config
    

class HDropout(keras.layers.Dropout):
    def __init__(self, rate: float):
        
        super().__init__(rate=rate)
        
    def build(self, input_shape: list):
        self.noise_shape = list(input_shape)
        self.noise_shape[4] = 1 # set both real and imaginary values to zero
        
        super().build(input_shape)
        
        
class HNonLin(keras.Layer):
    def __init__(self, fn: Callable = keras.activations.relu, bias_initializer: keras.Initializer = 'zero', 
                 bias_regularizer: keras.Regularizer = None, eps: float = 1e-4, name: str = 'HNonLin'):
        """The Harmonic Non Linearity Layer applies a pointwise non linearity on the magnitudes of the input after adding
        a bias without changing the shape.
        
        The input must have shape [batch_size, h, w, in_orders, complex, in_channels]. The output has the same shape.

        Args:
            fn (Callable, optional): The nonlinearity applied to the result. MUST map to non-negative reals R+.
                Defaults to keras.activations.relu.
            bias_initializer (Initializer, optional): The initializer used to initialize the bias. Defaults to 'zero'.
            bias_regularizer (Regularizer, optional): The regularzation applied to the bias. Defaults to None.
            eps (float, optional): Regularization since gradient of magnitudes is infinite at zero. Defaults to 1e-4.
            name (str, optional): The name of the layer. Defaults to 'HNonLin'.
        """
        super().__init__()
        self.fn = fn
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.eps = eps
        self.name = name

    def build(self, input_shape: list):
        """Initializes the bias.

        Args:
            input_shape (list): The shape of the input.
        """
        orders = input_shape[3]
        input_channels = input_shape[5]
        
        # separate bias for each planar feature map
        self.b = self.add_weight(shape=[1,1,1,orders,1,input_channels], name='b'+self.name, 
                                 initializer=self.bias_initializer, regularizer=self.bias_regularizer)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Applies a nonlinearity to the magnitudes of the input.

        Args:
            inputs (tf.Tensor): The input tensor.
            training (bool, optional): Whether in training or evaluation mode. Defaults to False.

        Returns:
            tf.Tensor: The output tensor of the same shape as the input.
        """
        with tf.name_scope(self.name) as scope:
            return h_ops.h_nonlin(inputs, self.fn, self.b, self.eps)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'fn': self.fn,
            'bias_initializer': self.bias_initializer,
            'bias_regularizer': self.bias_regularizer,
            'eps': self.eps,
            'name': self.name
        })
        return config
    
    
class MeanPool(keras.Layer):
    def __init__(self, ksize: tuple = (2,2), strides: tuple = (2,2), padding: str = 'VALID', name: str = 'MeanPool'):
        """The Mean Pooling Layer performs mean pooling on each planar feature map.
        
        The input must have shape [batch_size, h, w, in_orders, complex, in_channels] and outputs a tensor of the same
        format: [batch_size, h', w', in_orders, complex, in_channels].

        Args:
            ksize (tuple, optional): The size of the pooling window. Defaults to (2,2).
            strides (tuple, optional): The stride of the pooling window. Defaults to (2,2).
            padding (str, optional): Padding of the pool operation ('VALID' or 'SAME'). Defaults to 'VALID'.
            name (str, optional): The name of the layer. Defaults to 'MeanPool'.
        """
        super().__init__()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.name = name
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs mean pooling on the input.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor with the same dimensions as the input.
        """
        with tf.name_scope(self.name) as scope:
            return h_ops.mean_pooling(inputs, self.ksize, self.strides, self.padding)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'ksize': self.ksize,
            'strides': self.strides,
            'padding': self.padding,
            'name': self.name
        })
        return config
        
        
class Magnitudes(keras.Layer):
    def __init__(self, eps: float = 1e-12, keepdims: bool = False, name: str = 'Magnitudes'):
        """The Magnitudes Layer produces real feature maps by taking the magnitudes.
        
        The input must have shape [batch_size, h, w, in_orders, complex, in_channels]. By default the
        output has the shape [batch_size, h, w, in_orders, in_channels]. If `keepdims=True` the output shape is
        [batch_size, h, w, in_orders, 1, in_channels].

        Args:
            eps (float, optional): Regularization since gradient of magnitudes is infinite at zero. Defaults to 1e-12.
            keepdims (bool, optional): Whether to keep the spatial dimensions. Defaults to False.
            name (str, optional): The name of the layer. Defaults to 'Magnitudes'.
        """
        super().__init__()
        self.eps = eps
        self.keepdims = keepdims
        self.name = name
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calculates the magnitudes.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor either of shape [batch_size, h, w, in_orders, in_channels] or 
                [batch_size, h, w, in_orders, 1, in_channels] if `keepdims=True`
        """
        with tf.name_scope(self.name) as scope:
            return h_ops.stack_magnitudes(inputs, self.eps, self.keepdims)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'eps': self.eps,
            'keepdims': self.keepdims,
            'name': self.name
        })
        return config
        
    
class Reduce(keras.Layer):
    mean = tf.reduce_mean
    max = tf.reduce_max
    
    def __init__(self, fn: Callable, keepdims: bool = False, name: str = 'Reduce'):
        """The Reduce Layer reduces all dimensions of the input except the channels resulting in a single layer for each channel.
        
        By default the output shape is [batch_size, in_channels]. If `keepdims=True`, the reduced dimensions are not
        collapsed and have a size of 1.

        Args:
            fn (Callable): The permutation invariant pooling function such as `tf.reduce_max` or `tf.reduce_mean`.
            keepdims (bool, optional): Whether to keep the spatial dimensions. Defaults to False.
            name (str, optional): The name of the layer. Defaults to 'Reduce'.
        """
        super().__init__()
        self.fn = fn
        self.keepdims = keepdims
        self.name = name
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Reduces all dimensions except the channels dimension.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor either of shape [batch_size, in_channels] or with non-collapsed dimensions.
        """
        with tf.name_scope(self.name) as scope:
            dim = len(inputs.shape)
            reduce_axis = range(1, dim-1)
            return self.fn(inputs, axis=reduce_axis, keepdims=self.keepdims)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'fn': self.fn,
            'keepdims': self.keepdims,
            'name': self.name
        })
        return config
        