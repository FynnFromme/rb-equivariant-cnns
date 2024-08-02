"""
SE(2,N)-CNN Layers
"""

from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .ops import se2_conv


class LiftSE2NConv(keras.Layer):
    def __init__(self, channels: int, orientations: int, ksize: int, use_bias: bool = True, strides: tuple = (1, 1), 
                 padding: str = 'VALID', filter_initializer: keras.Initializer = None, 
                 bias_initializer: keras.Initializer = None, filter_regularizer: keras.Regularizer = None, 
                 bias_regularizer: keras.Regularizer = None, name: str = 'LiftConv'):
        """A Lifting Convolutional Layer convolves the Z2 input with rotated copies of the Z2 filters and thus 
        resulting in a SE(2,N) feature map with one rotation channel for each filter rotation.
        
        The input must have the shape [batch_size, h, w, in_channels]. The resulting output has an extra rotation
        dimension and thus a shape of [batch_size, h', w', orientations, channels].

        Args:
            channels (int): The number of output channels.
            orientations (int): The number of rotated filters.
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
        assert ksize%2==1, 'even filter sizes not supported' # rotation only implemented for odd sizes
        
        super().__init__()
        self.channels = channels
        self.orientations = orientations
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
            input_shape (list): The shape of the input. Must be of the form [batch_size, h, w, in_channels].
        """
        assert len(input_shape) == 4, 'transformation dimension not allowed yet'
        
        filter_shape = (self.ksize, self.ksize) + input_shape[-1:] + (self.channels,)
        self.filters = self.add_weight(name=self.name+'_w', dtype=tf.float32, shape=filter_shape,
                                       initializer=self.filter_initializer, regularizer=self.filter_regularizer)
        
        if self.use_bias:
            shape = [1, 1, 1, 1, self.channels]
            self.bias = self.add_weight(name=self.name+'_b', dtype=tf.float32, shape=shape,
                                        initializer=self.bias_initializer, regularizer=self.bias_regularizer)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs the Lifting Convolution on the input.

        Args:
            inputs (tf.Tensor): The input tensor of shape [batch_size, h, w, in_channels].

        Returns:
            tf.Tensor: The resulting tensor of shape [batch_size, h', w', orientations, channels].
        """
        with tf.name_scope(self.name) as scope:
            out = se2_conv.z2_se2n(inputs, self.filters, orientations=self.orientations, 
                                   strides=self.strides, padding=self.padding, name=self.name)[0]
            if self.use_bias:
                out = tf.add(out, self.bias)
            return out
            
    def get_transformed_filters(self) -> tf.Tensor:
        """Calculates the transformed filters based on the currently learned weights.

        Returns:
            tf.Tensor: The transformed filters of shape 
                [orientations, filter_size, filter_size, 1, in_channels, channels].
        """
        weights = self.get_weights()[0]
        filters = se2_conv.rotate_lifting_filters(weights, self.orientations)
        filters = tf.expand_dims(filters, axis=3)
        return filters
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'orientations': self.orientations,
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
        
        
class SE2NConv(keras.Layer):
    def __init__(self, channels: int, ksize: int, use_bias: bool = True, strides: tuple = (1, 1), 
                 padding: str = 'VALID', filter_initializer: keras.Initializer = None, 
                 bias_initializer: keras.Initializer = None, filter_regularizer: keras.Regularizer = None, 
                 bias_regularizer: keras.Regularizer = None, name: str = 'SE2NConv'):
        """A SE(2,N) Convolutional Layer convolves the SE(2,N) input with rotated copies of the SE(2,N) 
        filters and thus resulting in a SE(2,N) feature map with one rotation channel for each filter rotation.
        
        The input must have the shape [batch_size, h, w, orientations, in_channels]. The resulting output has an 
        extra rotation dimension and thus a shape of [batch_size, h', w', orientations, channels].

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
            name (str, optional): The name of the layer. Defaults to 'SE2NConv'.
        """
        assert ksize%2==1, 'even filter sizes not supported' # rotation only implemented for odd sizes
        
        super().__init__()
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
            input_shape (list): The shape of the input. Must be of the form [batch_size, h, w, orientations, in_channels].
        """
        assert len(input_shape) == 5, 'missing transformation dimension'
        
        filter_shape = (self.ksize, self.ksize) + input_shape[-2:] + (self.channels,)

        self.filters = self.add_weight(name=self.name+'_w', dtype=tf.float32, shape=filter_shape,
                                       initializer=self.filter_initializer, regularizer=self.filter_regularizer)
        
        if self.use_bias:
            shape = [1, 1, 1, 1, self.channels]
            self.bias = self.add_weight(name=self.name+'_b', dtype=tf.float32, shape=shape,
                                        initializer=self.bias_initializer, regularizer=self.bias_regularizer)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs the Lifting Convolution on the input.

        Args:
            inputs (tf.Tensor): The input tensor of shape [batch_size, h, w, orientations, in_channels].

        Returns:
            tf.Tensor: The resulting tensor of shape [batch_size, h', w', orientations, channels].
        """
        with tf.name_scope(self.name) as scope:
            out = se2_conv.se2n_se2n(inputs, self.filters, strides=self.strides, 
                                     padding=self.padding, name=self.name)[0]
            if self.use_bias:
                out = tf.add(out, self.bias)
            return out
            
    def get_transformed_filters(self) -> tf.Tensor:
        """Calculates the transformed filters based on the currently learned weights.

        Returns:
            tf.Tensor: The transformed filters of shape 
                [orientations, filter_size, filter_size, orientations, in_channels, channels].
        """
        weights = self.get_weights()[0]
        filters = se2_conv.rotate_gconv_filters(weights)
        return filters
    
    def get_config(self):
        config = super().get_config()
        config.update({
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
    
    
class LiftDNConv(keras.Layer):
    def __init__(self, channels: int, orientations: int, ksize: int, use_bias: bool = True, strides: tuple = (1, 1), 
                 padding: str = 'VALID', filter_initializer: keras.Initializer = None, 
                 bias_initializer: keras.Initializer = None, filter_regularizer: keras.Regularizer = None, 
                 bias_regularizer: keras.Regularizer = None, name: str = 'LiftConv'):
        """A Lifting Convolutional Layer convolves the Z2 input with flipped/rotated copies of the Z2 filters and thus 
        resulting in a DN feature map with one transformation channel for each filter transformation.
        
        The input must have the shape [batch_size, h, w, in_channels]. The resulting output has an extra transformation
        dimension and thus a shape of [batch_size, h', w', 2*orientations, channels].

        Args:
            channels (int): The number of output channels.
            orientations (int): The number of rotated filters.
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
        assert ksize%2==1, 'even filter sizes not supported' # rotation only implemented for odd sizes
        
        super().__init__()
        self.channels = channels
        self.orientations = orientations
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
            input_shape (list): The shape of the input. Must be of the form [batch_size, h, w, in_channels].
        """
        assert len(input_shape) == 4, 'transformation dimension not allowed yet'
        
        filter_shape = (self.ksize, self.ksize) + input_shape[-1:] + (self.channels,)
        self.filters = self.add_weight(name=self.name+'_w', dtype=tf.float32, shape=filter_shape,
                                       initializer=self.filter_initializer, regularizer=self.filter_regularizer)
        
        if self.use_bias:
            shape = [1, 1, 1, 1, self.channels]
            self.bias = self.add_weight(name=self.name+'_b', dtype=tf.float32, shape=shape,
                                        initializer=self.bias_initializer, regularizer=self.bias_regularizer)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs the Lifting Convolution on the input.

        Args:
            inputs (tf.Tensor): The input tensor of shape [batch_size, h, w, in_channels].

        Returns:
            tf.Tensor: The resulting tensor of shape [batch_size, h', w', 2*orientations, channels].
        """
        with tf.name_scope(self.name) as scope:
            out = se2_conv.z2_dn(inputs, self.filters, orientations=self.orientations, 
                                 strides=self.strides, padding=self.padding, name=self.name)[0]
            if self.use_bias:
                out = tf.add(out, self.bias)
            return out
            
    def get_transformed_filters(self) -> tf.Tensor:
        """Calculates the transformed filters based on the currently learned weights.

        Returns:
            tf.Tensor: The transformed filters of shape 
                [orientations, filter_size, filter_size, 1, in_channels, channels].
        """
        weights = self.get_weights()[0]
        filters = se2_conv.rotate_lifting_filters(weights, self.orientations)
        filters = se2_conv.flip_lifting_filters(filters)
        filters = tf.expand_dims(filters, axis=3)
        return filters
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'orientations': self.orientations,
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
        
        
class DNConv(keras.Layer):
    def __init__(self, channels: int, ksize: int, use_bias: bool = True, strides: tuple = (1, 1), 
                 padding: str = 'VALID', filter_initializer: keras.Initializer = None, 
                 bias_initializer: keras.Initializer = None, filter_regularizer: keras.Regularizer = None, 
                 bias_regularizer: keras.Regularizer = None, name: str = 'SE2NConv'):
        """A DN Convolutional Layer convolves the DN input with flipped/rotated copies of the DN
        filters and thus resulting in a DN feature map with one transformation channel for each filter transformation.
        
        The input must have the shape [batch_size, h, w, 2*orientations, in_channels]. The resulting output has an 
        extra transformation dimension and thus a shape of [batch_size, h', w', 2*orientations, channels].

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
            name (str, optional): The name of the layer. Defaults to 'SE2NConv'.
        """
        assert ksize%2==1, 'even filter sizes not supported' # rotation only implemented for odd sizes
        
        super().__init__()
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
            input_shape (list): The shape of the input. Must be of the form [batch_size, h, w, 2*orientations, in_channels].
        """
        assert len(input_shape) == 5, 'missing transformation dimension'
        
        filter_shape = (self.ksize, self.ksize) + input_shape[-2:] + (self.channels,)

        self.filters = self.add_weight(name=self.name+'_w', dtype=tf.float32, shape=filter_shape,
                                       initializer=self.filter_initializer, regularizer=self.filter_regularizer)
        
        if self.use_bias:
            shape = [1, 1, 1, 1, self.channels]
            self.bias = self.add_weight(name=self.name+'_b', dtype=tf.float32, shape=shape,
                                        initializer=self.bias_initializer, regularizer=self.bias_regularizer)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs the Lifting Convolution on the input.

        Args:
            inputs (tf.Tensor): The input tensor of shape [batch_size, h, w, 2*orientations, in_channels].

        Returns:
            tf.Tensor: The resulting tensor of shape [batch_size, h', w', 2*orientations, channels].
        """
        with tf.name_scope(self.name) as scope:
            out = se2_conv.dn_dn(inputs, self.filters, strides=self.strides, 
                                 padding=self.padding, name=self.name)[0]
            if self.use_bias:
                out = tf.add(out, self.bias)
            return out
            
    def get_transformed_filters(self) -> tf.Tensor:
        """Calculates the transformed filters based on the currently learned weights.

        Returns:
            tf.Tensor: The transformed filters of shape 
                [orientations, filter_size, filter_size, orientations, in_channels, channels].
        """
        weights = self.get_weights()[0]
        filters = se2_conv.rotate_gconv_filters(weights)
        filters = se2_conv.flip_gconv_filters(filters)
        return filters
    
    def get_config(self):
        config = super().get_config()
        config.update({
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
        """The Transformation Pooling Layer pools across transformation channels resulting in a single spatial feature map for 
        each channel, which is equivariant under transformation of the network input.
        
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
        """Pools across the transformations channels.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The resulting output tensor with a reduced transformations dimension.
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
        the channels and transformations dimension.

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
        """Checks the shape of the input. Must be of the form [batch_size, h, w, orientations, in_channels] or 
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
            # transform input to 2d feature maps
            in_shape = inputs.shape
            batch_size = tf.shape(inputs)[0] # batch size is unknown during construction, thus use tf.shape
            inputs = tf.reshape(inputs, tf.concat([[batch_size], inputs.shape[1:3], [np.prod(inputs.shape[3:])]], axis=0))
            
            # perform pooling on 2d feature maps
            outputs = tf.nn.pool(input=inputs, window_shape=self.ksize, pooling_type=self.pooling_type, 
                                 strides=self.strides, padding=self.padding, name=self.name)
            
            # transform back to 3d feature maps
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
        """The Reduce Spatial Layer reduces the spatial dimensions of the input resulting in a single layer for each 
        channel (and transformation channel).
        
        By default the output shape is [batch_size, orientations, in_channels]. If `keepdims=True`, the output 
        shape is [batch_size, 1, 1, orientations, in_channels]. This also works if the orientation dimension was 
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
      
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Reduces the spatial dimensions of the input.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor with reduced spatial dimensions.
        """
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