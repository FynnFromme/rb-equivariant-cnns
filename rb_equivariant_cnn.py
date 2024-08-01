import math

import tensorflow as tf
from tensorflow import keras

# TODO: Maybe share weights across some small vertical interval to save on parameters and assume it is translation invariant
#       for small translations -> especially for very high domains

# TODO: Custom DataAugmentation, BatchNorm, Dropout, SpatialPooling layers


class RB3D_Conv(keras.Layer):
    def __init__(self, channels: int, v_ksize: int, h_ksize: int, use_bias: bool = True, 
                 strides: tuple = (1, 1, 1), padding: str = 'REFLECT', filter_initializer: keras.Initializer = None, 
                 bias_initializer: keras.Initializer = None, filter_regularizer: keras.Regularizer = None, 
                 bias_regularizer: keras.Regularizer = None, name: str = 'RB3D_Conv'):
        """A 3D Rayleigh-Bénard Convolutional layer convolves the input with a 3d filter, while parameters
        are only shared horizontally since 3D Rayleigh-Bénard is not translation equivariant in the vertical 
        direction.
        
        The input must have the shape [batch_size, width, depth, height, in_channels]. 
        The resulting output has a shape of [batch_size, width', depth', height', channels].
        
        Currently only VALID padding is supported.

        Args:
            h_input (str): The group of input transformations.
            h_output (str):  The group of output transformations.
            channels (int): The number of output channels.
            v_ksize (int): The size of the filter in the vertical direction.
            h_ksize (int): The size of the filter in both horizontal directions.
            use_bias (bool): Whether to apply a bias to the output. The bias is leared independently for each channel 
                while being shared across transformation channels to ensure equivariance. Defaults to True.
            strides (tuple, optional): Stride used in the conv operation (width, depth, height). Defaults to (1, 1, 1).
            padding (str, optional): The padding used during convolution. Must be either 'REFLECT' (kernel wraps around
                the boarder), 'VALID' or 'SAME'. Padding is only used horizontally.
            filter_initializer (Initializer, optional): Initializer used to initialize the filters. Defaults to None.
            bias_initializer (Initializer, optional): Initializer used to initialize the bias. Defaults to None.
            filter_regularizer (Regularizer, optional): The regularzation applied to filter weights. Defaults to None.
            bias_regularizer (Regularizer, optional): The regularzation applied to the bias. Defaults to None.
            name (str, optional): The name of the layer. Defaults to 'RB3DGConv'.
        """
        super().__init__()
        
        assert padding in ('REFLECT', 'VALID', 'SAME')
        
        self.channels = channels
        self.v_ksize = v_ksize
        self.h_ksize = h_ksize
        self.use_bias = use_bias
        self.strides = strides
        self.padding = padding
        self.filter_initializer = filter_initializer
        self.bias_initializer = bias_initializer
        self.filter_regularizer = filter_regularizer
        self.bias_regularizer = bias_regularizer
        self.name = name
        
    def build(self, input_shape: list):
        """Initializes the filters of the layer.

        Args:
            input_shape (list): The input as shape [batch_size, width, depth, height, channels]
        """
        _, self.in_width, self.in_depth, self.in_height, self.in_channels = input_shape
        self.out_height = math.ceil((self.in_height - (self.v_ksize-1))/self.strides[2])
        
        filter_shape = [self.h_ksize, self.h_ksize, self.v_ksize, self.in_channels, self.out_height, self.channels]
        
        self.filters = self.add_weight(name=self.name+'_w', dtype=tf.float32, shape=filter_shape,
                                       initializer=self.filter_initializer, regularizer=self.filter_regularizer)
        
        if self.use_bias:
            shape = [1, 1, 1, self.out_height, self.channels]
            self.bias = self.add_weight(name=self.name+'_b', dtype=tf.float32, shape=shape,
                                        initializer=self.bias_initializer, regularizer=self.bias_regularizer)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Convolves the input with a 3D filter, where weights are only shared horizontally.

        Args:
            input (tf.Tensor): The input to the layer of shape [batch_size, width, depth, height, channels].

        Returns:
            tf.Tensor: The output of the layer of shape [batch_size, out_width, out_depth, out_height, out_channels].
        """
        
        batch_size = tf.shape(inputs)[0] # batch size is unknown during construction, thus use tf.shape
        
        padded_filters = self.pad_filters(self.filters)
        
        inputs_reshaped = tf.reshape(inputs, [batch_size, 
                                              self.in_width, self.in_depth, 
                                              self.in_height*self.in_channels])
        filters_reshaped = tf.reshape(padded_filters, [self.h_ksize, self.h_ksize, 
                                                       self.in_height*self.in_channels,
                                                       self.out_height*self.channels])
        
        inputs_reshaped = self.pad_inputs(inputs_reshaped)
        
        # conv2d output has shape [batch_size, out_width, out_depth, out_height*out_channels]
        output_reshaped = tf.nn.conv2d(inputs_reshaped, filters_reshaped, strides=self.strides[:2], padding='VALID')
        
        output = tf.reshape(output_reshaped, output_reshaped.shape[:3] + [self.out_height, self.channels])
        
        return output
        
    def pad_filters(self, filters: tf.Tensor) -> tf.Tensor:
        """Pads the 3D convolution filters of height z with zeros to the height of the input data.

        Args:
            filters (tf.Tensor): The filter of shape [x, y, z, in_channels, out_height, out_channels].

        Returns:
            tf.Tensor: The padded filter of shape [x, y, in_height, in_channels, out_height, out_channels].
        """
        
        padding_shape = filters.shape[:2] + [self.in_height-self.v_ksize] + filters.shape[3:]
        padding = tf.zeros(padding_shape)
        
        # add padding below the filters
        padded_filters = tf.concat([filters, padding], axis=2)
        
        # roll the padded filter so that the 3D filters are at their respective heights
        padded_filters = tf.transpose(padded_filters, [4, 0, 1, 2, 3, 5]) # -> (out_height,x,y,h,c_in,c_out)
        rolled_filters = tf.map_fn(
            lambda x: tf.roll(x[0], shift=x[1]*self.strides[2], axis=2), # roll along the h axis of the filter
            elems=[padded_filters, tf.range(0, self.out_height)],
            fn_output_signature=padded_filters.dtype
        )
        rolled_filters = tf.transpose(rolled_filters, [1, 2, 3, 4, 0, 5]) # -> (x,y,h,c_in,out_height,c_out)
        
        return rolled_filters
    
    def pad_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        """Adds padding to the input data of the layer.

        Args:
            inputs (tf.Tensor): Must be of shape [batch_size, width, depth, height*in_channels]

        Returns:
            tf.Tensor: The padded input of shape [batch_size, width', depth', height*in_channels]
        """
        if self.padding == 'VALID':
            return inputs
        
        width_padding = required_padding(self.h_ksize, self.in_width, self.strides[0])
        depth_padding = required_padding(self.h_ksize, self.in_depth, self.strides[1])
        # batch, width, depth, height*channel
        padding = [[0, 0], width_padding, depth_padding, [0, 0]] 
        return tf.pad(inputs, padding, mode='CONSTANT' if self.padding == 'SAME' else self.padding)
        
      
def required_padding(ksize: int, input_size: int, stride: int) -> tuple:
    """Calculates the required padding for a dimension using SAME padding.

    Args:
        ksize (int): The kernel size in the dimension.
        input_size (int): The input size in the dimension.
        stride (int): The stride in the dimension.

    Returns:
        tuple: A tuple containing the padding on the left/bottom and right/top.
    """
    if ksize % stride == 0:
        padding_needed = max(ksize-stride, 0)
    else:
        padding_needed = max(ksize-(input_size%stride), 0)

    return padding_needed//2, math.ceil(padding_needed/2)