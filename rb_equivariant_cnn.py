import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

# TODO: Maybe share weights across some small vertical interval to save on parameters and assume it is translation invariant
#       for small translations -> especially for very high domains

# TODO: Use Interpolation in upsampling
# TODO: Maybe link Decoder upsamling to Encoder upsampling (ask Jason)


class RB3D_Conv(keras.Layer):
    def __init__(self, channels: int, h_ksize: int, v_ksize: int, use_bias: bool = True, 
                 strides: tuple = (1, 1, 1), h_padding: str = 'VALID', v_padding: str = 'VALID', filter_initializer: keras.Initializer = None, 
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
            h_ksize (int): The size of the filter in both horizontal directions.
            v_ksize (int): The size of the filter in the vertical direction.
            use_bias (bool): Whether to apply a bias to the output. The bias is leared independently for each channel 
                while being shared across transformation channels to ensure equivariance. Defaults to True.
            strides (tuple, optional): Stride used in the conv operation (width, depth, height). Defaults to (1, 1, 1).
            h_padding (str, optional): The horizontal padding used during convolution in width and depth direction.
                Must be either 'WRAP' (kernel wraps around the boarder to the opposite side), 'VALID' or 'SAME'. Defaults to 'VALID'.
            v_padding (str, optional): The vertical padding used during convolution in height direction.
                Must be either 'VALID' or 'SAME'. Defaults to 'VALID'.
            filter_initializer (Initializer, optional): Initializer used to initialize the filters. Defaults to None.
            bias_initializer (Initializer, optional): Initializer used to initialize the bias. Defaults to None.
            filter_regularizer (Regularizer, optional): The regularzation applied to filter weights. Defaults to None.
            bias_regularizer (Regularizer, optional): The regularzation applied to the bias. Defaults to None.
            name (str, optional): The name of the layer. Defaults to 'RB3DGConv'.
        """
        super().__init__()
        
        assert h_padding in ('WRAP', 'VALID', 'SAME')
        assert v_padding in ('VALID', 'SAME')
        
        self.channels = channels
        self.h_ksize = h_ksize
        self.v_ksize = v_ksize
        self.use_bias = use_bias
        self.strides = strides
        self.h_padding = h_padding
        self.v_padding = v_padding
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
        
        self.padded_in_height = self.in_height + (sum(required_padding(self.v_ksize, self.in_height, self.strides[2])) if self.v_padding != 'VALID' else 0)
        # self.out_height = math.ceil((self.in_height - (self.v_ksize-1))/self.strides[2]) # output height without padding
        self.out_height = (self.padded_in_height - self.v_ksize)//self.strides[2] + 1 # TODO verify this formula
        
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
        
        padded_filters = self.pad_filters(self.filters) # converts 3d kernels to 2d kernels over full height
        
        padded_inputs = self.pad_inputs(inputs) # add conventional padding to inputs
        
        # combine height and channel dimensions
        inputs_reshaped = tf.reshape(padded_inputs, padded_inputs.shape[:3]+[np.prod(padded_inputs.shape[-2:])])
        filters_reshaped = tf.reshape(padded_filters, [self.h_ksize, self.h_ksize, 
                                                       self.padded_in_height*self.in_channels,
                                                       self.out_height*self.channels])
        
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
        
        padding_shape = filters.shape[:2] + [self.padded_in_height-self.v_ksize] + filters.shape[3:]
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
            inputs (tf.Tensor): Must be of shape [batch_size, width, depth, height, in_channels]

        Returns:
            tf.Tensor: The padded input of shape [batch_size, width', depth', height', in_channels]
        """
        if self.h_padding == 'SAME':        
            width_padding = required_padding(self.h_ksize, self.in_width, self.strides[0])
            depth_padding = required_padding(self.h_ksize, self.in_depth, self.strides[1])
            # batch, width, depth, height, channel
            padding = [[0, 0], width_padding, depth_padding, [0, 0], [0, 0]] 
            inputs = tf.pad(inputs, padding, mode='CONSTANT')
            
        if self.h_padding == 'WRAP':
            width_padding = required_padding(self.h_ksize, self.in_width, self.strides[0])
            depth_padding = required_padding(self.h_ksize, self.in_depth, self.strides[1])
            
            inputs = tf.concat([inputs[:,-width_padding[0]:], inputs, inputs[:,:width_padding[1]]], axis=1)
            inputs = tf.concat([inputs[:,:,-depth_padding[0]:], inputs, inputs[:,:,:depth_padding[1]]], axis=2)
            
        if self.v_padding == 'SAME':
            height_padding = required_padding(self.v_ksize, self.in_height, self.strides[2])
            # batch, width, depth, height, channel
            padding = [[0, 0], [0, 0], [0, 0], height_padding, [0, 0]]
            inputs = tf.pad(inputs, padding, mode='CONSTANT')
        return inputs
        
      
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


class SpatialPooling(keras.Layer):
    def __init__(self, ksize: tuple = (2,2,2), strides: tuple = (2,2,2), pooling_type: str = 'MAX',
                 padding: str = 'VALID', name: str = 'SpatialPooling'):
        """The Spatial Pooling Layer performs spatial pooling on each 3D feature map.
        
        Input and output have shape [batch_size, width, depth, height, channels].

        Args:
            ksize (tuple, optional): The size of the pooling window. Defaults to (2,2,2).
            strides (tuple, optional): The stride of the pooling window. Defaults to (2,2,2).
            pooling_type (str, optional): Whether to use 'MAX' or 'AVG' pooling. Defaults to 'MAX'.
            padding (str, optional): Padding of the pool operation ('VALID' or 'SAME'). Defaults to 'VALID'.
            name (str, optional): The name of the layer. Defaults to 'SpatialPooling'.
        """
        super().__init__()
        self.ksize = ksize
        self.strides = strides
        self.pooling_type = pooling_type
        self.padding = padding
        self.name = name
       
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Applies spatial pooling to each 3D feature map of the input.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The pooled output tensor.
        """
        with tf.name_scope(self.name) as scope:            
            outputs = tf.nn.pool(input=inputs, window_shape=self.ksize, 
                                 pooling_type=self.pooling_type, strides=self.strides,
                                 padding=self.padding, name=self.name)
            
            return outputs
        

# TODO: Maybe link Decoder upsamling to Encoder upsampling (ask Jason)
# TODO: use linear interpolation
UpSampling = keras.layers.UpSampling3D


class BatchNorm(keras.layers.BatchNormalization):
    def __init__(self, momentum: float = 0.99, epsilon: float = 0.001, name: str = 'BatchNorm', **kwargs):
        """The Batch Normalization Layer applies batch normalization to the input. The gamma and beta parameters are
        seperately learned for each channel as well as height.
        
        Input and output have shape [batch_size, width, depth, height, channels].

        Args:
            momentum (float, optional): Momentum for the moving average. Defaults to 0.99.
            epsilon (float, optional): Small float added to the variance to avoid dividing by zero. Defaults to 0.001.
            name (str, optional): The name of the layer. Defaults to 'BatchNorm'.
        """
        super().__init__(axis=-1, momentum=momentum, epsilon=epsilon, name=name, **kwargs)
        
    def build(self, input_shape: list):
        # give the subclass the shape of the transformed input
        batch_size, width, depth, height, channels = input_shape
        super().build([batch_size, width, depth, height*channels])
        self.input_spec = keras.InputSpec(ndim=5) # overwrite input spec set it super().build
    
    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Apples batch normalization to the data.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The normalized data of shape [batch_size, width, depth, height, channels].
        """
        with tf.name_scope(self.name) as scope:
            in_height, in_channels = inputs.shape[-2:]
            batch_size = tf.shape(inputs)[0] # batch size is unknown during construction, thus use tf.shape
            
            # bring data into shape (batch_size, width, depth, height*channel)
            inputs = tf.reshape(inputs, tf.concat([[batch_size], inputs.shape[1:3], [np.prod(inputs.shape[3:])]], axis=0))
            
            outputs = super().call(inputs, *args, **kwargs)

            # bring data back into shape (batch_size, width, depth, height, channel)
            outputs = tf.reshape(outputs, tf.concat([[batch_size], outputs.shape[1:3], [in_height, in_channels]], axis=0))
            
            return outputs