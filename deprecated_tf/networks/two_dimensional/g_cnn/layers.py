from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .ops.gconv import splitgconv2d
from .ops.gconv.transform_filter import transform_filter_2d_nhwc

# TODO: Test which data shape is faster. [batch,h,w,transformation,channel] or [batch,h,w,transformation*channel]

class GConv(keras.Layer):
    def __init__(self, h_input: str, h_output: str, channels: int, ksize: int, use_bias: bool = True, strides: tuple = (1, 1), 
                 padding: str = 'VALID', filter_initializer: keras.Initializer = None, 
                 bias_initializer: keras.Initializer = None, filter_regularizer: keras.Regularizer = None, 
                 bias_regularizer: keras.Regularizer = None, name: str = 'GConv'):
        """A G-Convolutional Layer convolves the input of type `h_input` with transformed copies of the `h_input` 
        filters and thus resulting in a `h_output` feature map with one transformation channel for each filter transformation.
        
        The input must have the shape [batch_size, h, w, in_transformations, in_channels]. The resulting output 
        has a shape of [batch_size, h', w', out_transformations, channels].

        Args:
            h_input (str): The group of input transformations.
            h_output (str):  The group of output transformations.
            channels (int): The number of output channels.
            ksize (int): The size of the filter.
            use_bias (bool): Whether to apply a bias to the output. The bias is leared independently for each channel 
                while being shared across transformation channels to ensure equivariance. Defaults to True.
            strides (tuple, optional): Stride used in the conv operation. Defaults to (1, 1).
            padding (str, optional): Padding of the conv operation ('VALID' or 'SAME'). Defaults to 'VALID'.
            filter_initializer (Initializer, optional): Initializer used to initialize the filters. Defaults to None.
            bias_initializer (Initializer, optional): Initializer used to initialize the bias. Defaults to None.
            filter_regularizer (Regularizer, optional): The regularzation applied to filter weights. Defaults to None.
            bias_regularizer (Regularizer, optional): The regularzation applied to the bias. Defaults to None.
            name (str, optional): The name of the layer. Defaults to 'GConv'.
        """
        super().__init__()
        self.h_input = h_input
        self.h_output = h_output
        self.channels = channels
        self.ksize = ksize
        self.use_bias = use_bias
        self.strides = strides
        self.padding = padding
        self.filter_initializer = filter_initializer
        self.bias_initializer = bias_initializer
        self.filter_regularizer = filter_regularizer
        self.bias_regularizer = bias_regularizer
        self.name = name
        
    def build(self, input_shape: list):
        """Initializes the filter weights.

        Args:
            input_shape (list): The shape of the input. Must be of the form [batch_size, h, w, in_transformations, in_channels].
        """
        assert len(input_shape) == 5, 'missing transformation dimension'
        assert input_shape[-2] == {'Z2': 1, 'C4': 4, 'D4': 8}[self.h_input], 'wrong number of transformation channels'
        
        in_channels = input_shape[-1]
        self.gconv_indices, self.gconv_shape_info, filter_shape = splitgconv2d.gconv2d_util(
            h_input=self.h_input, h_output=self.h_output, in_channels=in_channels, 
            out_channels=self.channels, ksize=self.ksize)
        
        self.filters = self.add_weight(name=self.name+'_w', dtype=tf.float32, shape=filter_shape,
                                       initializer=self.filter_initializer, regularizer=self.filter_regularizer)
        
        if self.use_bias:
            shape = [1, 1, 1, 1, self.channels]
            self.bias = self.add_weight(name=self.name+'_b', dtype=tf.float32, shape=shape,
                                        initializer=self.bias_initializer, regularizer=self.bias_regularizer)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs the G-Convolution on the input.

        Args:
            inputs (tf.Tensor): The input tensor of shape [batch_size, h, w, in_transformations, in_channels].

        Returns:
            tf.Tensor: The resulting tensor of shape [batch_size, h', w', out_transformations, channels].
        """
        with tf.name_scope(self.name) as scope:
            out = splitgconv2d.gconv2d(input=inputs, filters=self.filters, strides=self.strides, 
                                       padding=self.padding, gconv_indices=self.gconv_indices, 
                                       gconv_shape_info=self.gconv_shape_info, name=self.name)
            if self.use_bias:
                out = tf.add(out, self.bias)
            return out
        
    def get_transformed_filters(self) -> tf.Tensor:
        """Calculates the transformed filters based on the currently learned weights.

        Returns:
            tf.Tensor: The transformed filters of shape 
                [out_transformations, filter_size, filter_size, in_transformations, in_channels, channels].
        """
        no, nto, ni, nti, n = self.gconv_shape_info
        filters = transform_filter_2d_nhwc(w=self.get_weights()[0], flat_indices=self.gconv_indices, 
                                           shape_info=self.gconv_shape_info)
        filters = tf.reshape(filters, [n, n, nti, ni, nto, no])
        filters = tf.transpose(filters, [4, 0, 1, 2, 3, 5])
        return filters
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'h_input': self.h_input,
            'h_output': self.h_output,
            'channels': self.channels,
            'ksize': self.ksize,
            'use_bias': self.use_bias,
            'strides': self.strides,
            'padding': self.padding,
            'filter_initializer': self.filter_initializer,
            'bias_initializer': self.bias_initializer,
            'filter_regularizer': self.filter_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'name': self.name
        })
        return config
            

class LiftP4Conv(GConv):
    def __init__(self, channels: int, ksize: int, use_bias: bool = True, strides: tuple = (1, 1), 
                 padding: str = 'VALID', filter_initializer: keras.Initializer = None, 
                 bias_initializer: keras.Initializer = None, filter_regularizer: keras.Regularizer = None, 
                 bias_regularizer: keras.Regularizer = None, name: str = 'LiftConv'):
        """A P4 Lifting Convolutional Layer convolves the Z2 input with 90° rotated copies of the Z2 filters 
        and thus resulting in a P4 feature map with one rotation channel for each filter rotation.
        
        The input must have the shape [batch_size, h, w, 1, in_channels]. The resulting output has a shape of 
        [batch_size, h', w', 4, channels].

        Args:
            channels (int): The number of output channels.
            ksize (int): The size of the filter.
            use_bias (bool): Whether to apply a bias to the output. The bias is leared independently for each channel 
                while being shared across transformation channels to ensure equivariance. Defaults to True.
            strides (tuple, optional): Stride used in the conv operation. Defaults to (1, 1).
            padding (str, optional): Padding of the conv operation ('VALID' or 'SAME'). Defaults to 'VALID'.
            filter_initializer (Initializer, optional): Initializer used to initialize the filters. Defaults to None.
            bias_initializer (Initializer, optional): Initializer used to initialize the bias. Defaults to None.
            filter_regularizer (Regularizer, optional): The regularzation applied to filter weights. Defaults to None.
            bias_regularizer (Regularizer, optional): The regularzation applied to the bias. Defaults to None.
            name (str, optional): The name of the layer. Defaults to 'LiftConv'.
        """
        super().__init__('Z2', 'C4', channels, ksize, use_bias, strides, padding, 
                         filter_initializer, bias_initializer, filter_regularizer, bias_regularizer, name)
        
        
class LiftP4MConv(GConv):
    def __init__(self, channels: int, ksize: int, use_bias: bool = True, strides: tuple = (1, 1), 
                 padding: str = 'VALID', filter_initializer: keras.Initializer = None, 
                 bias_initializer: keras.Initializer = None, filter_regularizer: keras.Regularizer = None, 
                 bias_regularizer: keras.Regularizer = None, name: str = 'LiftConv'):
        """A P4M Lifting Convolutional Layer convolves the Z2 input with 90° rotated and/or flipped copies of the 
        Z2 filters and thus resulting in a P4M feature map with one transformation channel for each filter rotation-flip.
        
        The input must have the shape [batch_size, h, w, 1, in_channels]. The resulting output has a shape of 
        [batch_size, h', w', 8, channels].

        Args:
            channels (int): The number of output channels.
            ksize (int): The size of the filter.
            use_bias (bool): Whether to apply a bias to the output. The bias is leared independently for each channel 
                while being shared across transformation channels to ensure equivariance. Defaults to True.
            strides (tuple, optional): Stride used in the conv operation. Defaults to (1, 1).
            padding (str, optional): Padding of the conv operation ('VALID' or 'SAME'). Defaults to 'VALID'.
            filter_initializer (Initializer, optional): Initializer used to initialize the filters. Defaults to None.
            bias_initializer (Initializer, optional): Initializer used to initialize the bias. Defaults to None.
            filter_regularizer (Regularizer, optional): The regularzation applied to filter weights. Defaults to None.
            bias_regularizer (Regularizer, optional): The regularzation applied to the bias. Defaults to None.
            name (str, optional): The name of the layer. Defaults to 'LiftConv'.
        """
        super().__init__('Z2', 'D4', channels, ksize, use_bias, strides, padding, 
                         filter_initializer, bias_initializer, filter_regularizer, bias_regularizer, name)
       
        
class P4Conv(GConv):
    def __init__(self, channels: int, ksize: int, use_bias: bool = True, strides: tuple = (1, 1), 
                 padding: str = 'VALID', filter_initializer: keras.Initializer = None, 
                 bias_initializer: keras.Initializer = None, filter_regularizer: keras.Regularizer = None, 
                 bias_regularizer: keras.Regularizer = None, name: str = 'P4Conv'):
        """A P4 Lifting Convolutional Layer convolves the P4 input with 90° rotated copies of the P4 filters and 
        thus resulting in a P4 feature map with one rotation channel for each filter rotation.
        
        The input must have the shape [batch_size, h, w, 4, in_channels]. The resulting output has a shape of 
        [batch_size, h', w', 4, channels].

        Args:
            channels (int): The number of output channels.
            ksize (int): The size of the filter.
            use_bias (bool): Whether to apply a bias to the output. The bias is leared independently for each channel 
                while being shared across transformation channels to ensure equivariance. Defaults to True.
            strides (tuple, optional): Stride used in the conv operation. Defaults to (1, 1).
            padding (str, optional): Padding of the conv operation ('VALID' or 'SAME'). Defaults to 'VALID'.
            filter_initializer (Initializer, optional): Initializer used to initialize the filters. Defaults to None.
            bias_initializer (Initializer, optional): Initializer used to initialize the bias. Defaults to None.
            filter_regularizer (Regularizer, optional): The regularzation applied to filter weights. Defaults to None.
            bias_regularizer (Regularizer, optional): The regularzation applied to the bias. Defaults to None.
            name (str, optional): The name of the layer. Defaults to 'P4Conv'.
        """
        super().__init__('C4', 'C4', channels, ksize, use_bias, strides, padding, 
                         filter_initializer, bias_initializer, filter_regularizer, bias_regularizer, name)
        
        
class P4MConv(GConv):
    def __init__(self, channels: int, ksize: int, use_bias: bool = True, strides: tuple = (1, 1), 
                 padding: str = 'VALID', filter_initializer: keras.Initializer = None, 
                 bias_initializer: keras.Initializer = None, filter_regularizer: keras.Regularizer = None, 
                 bias_regularizer: keras.Regularizer = None, name: str = 'P4MConv'):
        """A P4M Lifting Convolutional Layer convolves the P4M input with 90° rotated and/or flipped copies of the 
        P4M filters and thus resulting in a P4M feature map with one transformation channel for each filter rotation-flip.
        
        The input must have the shape [batch_size, h, w, 8, in_channels]. The resulting output has a shape of 
        [batch_size, h', w', 8, channels].

        Args:
            channels (int): The number of output channels.
            ksize (int): The size of the filter.
            use_bias (bool): Whether to apply a bias to the output. The bias is leared independently for each channel 
                while being shared across transformation channels to ensure equivariance. Defaults to True.
            strides (tuple, optional): Stride used in the conv operation. Defaults to (1, 1).
            padding (str, optional): Padding of the conv operation ('VALID' or 'SAME'). Defaults to 'VALID'.
            filter_initializer (Initializer, optional): Initializer used to initialize the filters. Defaults to None.
            bias_initializer (Initializer, optional): Initializer used to initialize the bias. Defaults to None.
            filter_regularizer (Regularizer, optional): The regularzation applied to filter weights. Defaults to None.
            bias_regularizer (Regularizer, optional): The regularzation applied to the bias. Defaults to None.
            name (str, optional): The name of the layer. Defaults to 'P4MConv'.
        """
        super().__init__('D4', 'D4', channels, ksize, use_bias, strides, padding, 
                         filter_initializer, bias_initializer, filter_regularizer, bias_regularizer, name)
    

class BatchNormalization(keras.layers.BatchNormalization):
    def __init__(self, momentum: float = 0.99, epsilon: float = 0.001, name: str = 'BatchNorm', *args, **kwargs):
        """The Batch Normalization Layer applies batch normalization to the input. The gamma and beta parameters are
        seperately learned for each input channel while being shared accross transformation channels to ensure equivariance.

        Args:
            momentum (float, optional): Momentum for the moving average. Defaults to 0.99.
            epsilon (float, optional): Small float added to the variance to avoid dividing by zero. Defaults to 0.001.
            name (str, optional): The name of the layer. Defaults to 'BatchNorm'.
        """
        super().__init__(-1, momentum, epsilon, *args, **kwargs)
        self.name = name


class TransformationPooling(keras.Layer):
    max = tf.reduce_max
    mean = tf.reduce_mean
    
    def __init__(self, fn: Callable, keepdims: bool = False, name: str = 'TransformationPooling'):
        """The Transformation Pooling Layer pools across transformation channels resulting in a single spatial 
        feature map for each channel, which is equivariant under transformation of the network input.
        
        By default the output shape is [batch_size, h, w, in_channels]. If `keepdims=True`, the output shape is 
        [batch_size, h, w, 1, in_channels]. This also works if the spatial dimension was reduced previously resulting
        in shape [batch_size, in_channels] and [batch_size, 1, in_channels] respectively.

        Args:
            fn (Callable): The permutation invariant pooling function such as `tf.reduce_max` or `tf.reduce_mean`.
            keepdims (bool, optional): Whether to keep the transformation dimension. Defaults to False.
            name (str, optional): The name of the layer. Defaults to 'TransformationPooling'.
        """
        super().__init__()
        self.fn = fn
        self.keepdims = keepdims
        self.name = name
        
    def build(self, input_shape: list):
        """Checks the shape of the input. Must be of the form [batch_size, h, w, transformations, in_channels] or 
        [batch_size, transformations, in_channels].

        Args:
            input_shape (list): The shape of the layer's input.
        """
        assert len(input_shape) in (3,5), 'missing transformation dimension' # 3 if spatial reduced
      
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Pools across the transformation channels.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The resulting output tensor with a reduced transformation dimension.
        """
        with tf.name_scope(self.name) as scope:
            return self.fn(inputs, axis=-2, keepdims=self.keepdims, name=self.name)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'fn': self.fn,
            'keepdims': self.keepdims,
            'name': self.name
        })
        return config


class SpatialPooling(keras.Layer):
    def __init__(self, ksize: tuple = (2,2), pooling_type: str = 'MAX', strides: tuple = (2,2),
                 padding: str = 'VALID', name: str = 'SpatialPooling'):
        """The Spatial Pooling Layer performs spatial pooling on each planar feature map across both
        the channels and transformation dimension.

        Args:
            ksize (tuple, optional): The size of the pooling window. Defaults to (2,2).
            pooling_type (str, optional): Whether to use 'MAX' or 'AVG' pooling. Defaults to 'MAX'.
            strides (tuple, optional): The stride of the pooling window. Defaults to (2,2).
            padding (str, optional): Padding of the pool operation ('VALID' or 'SAME'). Defaults to 'VALID'.
            name (str, optional): The name of the layer. Defaults to 'SpatialPooling'.
        """
        super().__init__()
        self.ksize = ksize
        self.pooling_type = pooling_type
        self.strides = strides
        self.padding = padding
        self.name = name
        
    def build(self, input_shape: list):
        """Checks the shape of the input. Must be of the form [batch_size, h, w, transformations, in_channels] or 
        [batch_size, h, w, in_channels].

        Args:
            input_shape (list): The shape of the layer's input.
        """
        assert len(input_shape) > 3, 'Spatial dimension reduced'
       
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Applies spatial pooling to each planar feature map of the input.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The pooled output tensor.
        """
        with tf.name_scope(self.name) as scope:
            in_shape = inputs.shape
            batch_size = tf.shape(inputs)[0] # batch size is unknown during construction, thus use tf.shape
            inputs = tf.reshape(inputs, tf.concat([[batch_size], inputs.shape[1:3], [np.prod(inputs.shape[3:])]], axis=0))
            
            outputs = tf.nn.pool(input=inputs, window_shape=self.ksize, 
                                 pooling_type=self.pooling_type, strides=self.strides,
                                 padding=self.padding, name=self.name)
            
            return tf.reshape(outputs, tf.concat([[batch_size], outputs.shape[1:3], in_shape[3:]], axis=0))
          
    def get_config(self):
        config = super().get_config()
        config.update({
            'ksize': self.ksize,
            'pooling_type': self.pooling_type,
            'strides': self.strides,
            'padding': self.padding,
            'name': self.name
        })
        return config

class ReduceSpatial(keras.Layer):
    max = tf.reduce_max
    mean = tf.reduce_mean
    
    def __init__(self, fn: Callable, keepdims: bool = False, name: str = 'ReduceSpatial'):
        """The Reduce Spatial Layer reduces the spatial dimensions of the input resulting in a single layer for 
        each channel (and transformation channel).
        
        By default the output shape is [batch_size, transformations, in_channels]. If `keepdims=True`, the output 
        shape is [batch_size, 1, 1, transformations, in_channels]. This also works if the transformations dimension was 
        reduced previously resulting in shape [batch_size, in_channels] and [batch_size, 1, 1, in_channels] respectively.

        Args:
            fn (Callable): The permutation invariant pooling function such as `tf.reduce_max` or `tf.reduce_mean`.
            keepdims (bool, optional): Whether to keep the spatial dimensions. Defaults to False.
            name (str, optional): The name of the layer. Defaults to 'ReduceSpatial'.
        """
        super().__init__()
        self.fn = fn
        self.keepdims = keepdims
        self.name = name
      
    def call(self, inputs):
        with tf.name_scope(self.name) as scope:
            return self.fn(inputs, axis=[1,2], keepdims=self.keepdims, name=self.name)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'fn': self.fn,
            'keepdims': self.keepdims,
            'name': self.name
        })
        return config