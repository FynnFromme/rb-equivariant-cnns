import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

from cnns_2d.g_cnn.layers import GConv
from cnns_2d.g_cnn.ops.gconv import splitgconv2d

# TODO: Maybe share weights across some small vertical interval to save on parameters and assume it is translation invariant
#       for small translations -> especially for very high domains

# TODO: Custom DataAugmentation, BatchNorm, Dropout, Spatial- and TransformationPooling and Upsampling layers


class RB3D_G_Conv(GConv):
    def __init__(self, h_input: str, h_output: str, channels: int, h_ksize: int, v_ksize: int, use_bias: bool = True, 
                 strides: tuple = (1, 1, 1), h_padding: str = 'VALID', v_padding: str = 'VALID', 
                 filter_initializer: keras.Initializer = None, bias_initializer: keras.Initializer = None, 
                 filter_regularizer: keras.Regularizer = None, bias_regularizer: keras.Regularizer = None, name: str = 'RB3D_G_Conv'):
        """A G-Convolutional Layer convolves the input of type `h_input` with transformed copies of the `h_input` 
        filters and thus resulting in a `h_output` feature map with one transformation channel for each filter transformation.
        
        The 3D Rayleigh Benard G-Convolutional layer extends this idea by using 3D filters, while parameters
        are only shared horizontally since 3D Rayleigh-Bénard is not translation equivariant in the vertical 
        direction. 
        
        The input must have the shape [batch_size, width, depth, in_transformations, height, in_channels]. 
        The resulting output has a shape of [batch_size, width', depth', out_transformations, height', channels].
        
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
                Must be either 'REFLECT' (kernel wraps around the boarder), 'VALID' or 'SAME'. Defaults to 'VALID'.
            v_padding (str, optional): The vertical padding used during convolution in height direction.
                Must be either 'VALID' or 'SAME'. Defaults to 'VALID'.
            filter_initializer (Initializer, optional): Initializer used to initialize the filters. Defaults to None.
            bias_initializer (Initializer, optional): Initializer used to initialize the bias. Defaults to None.
            filter_regularizer (Regularizer, optional): The regularzation applied to filter weights. Defaults to None.
            bias_regularizer (Regularizer, optional): The regularzation applied to the bias. Defaults to None.
            name (str, optional): The name of the layer. Defaults to 'RB3D_G_Conv'.
        """
        assert h_padding in ('REFLECT', 'VALID', 'SAME')
        assert v_padding in ('VALID', 'SAME')
        
        super().__init__(h_input=h_input, h_output=h_output, channels=channels, ksize=h_ksize, use_bias=use_bias, 
                         strides=strides, padding='VALID', filter_initializer=filter_initializer, 
                         bias_initializer=bias_initializer, filter_regularizer=filter_regularizer, 
                         bias_regularizer=bias_regularizer, name=name)
        self.h_ksize = h_ksize
        self.v_ksize = v_ksize
        self.h_padding = h_padding
        self.v_padding = v_padding
        
    def build(self, input_shape: list):
        """Initializes the filters of the layer.

        Args:
            input_shape (list): The input as shape [batch_size, width, depth, in_transformations, height, in_channels].
        """
        _, self.in_width, self.in_depth, self.in_transformations, self.in_height, self.in_channels = input_shape
        
        self.padded_in_height = self.in_height + (sum(required_padding(self.v_ksize, self.in_height, self.strides[2])) if self.v_padding != 'VALID' else 0)
        # self.out_height = math.ceil((self.in_height - (self.v_ksize-1))/self.strides[2]) # output height without padding
        self.out_height = (self.padded_in_height - self.v_ksize)//self.strides[2] + 1 # TODO verify this formula
        
        self.gconv_indices, self.gconv_shape_info, _ = splitgconv2d.gconv2d_util(
            h_input=self.h_input, h_output=self.h_output, in_channels=self.padded_in_height*self.in_channels, 
            out_channels=self.out_height*self.channels, ksize=self.h_ksize)
        
        filter_shape = [self.h_ksize, self.h_ksize, self.in_transformations, 
                        self.v_ksize, self.in_channels, self.out_height, self.channels]
        
        self.filters = self.add_weight(name=self.name+'_w', dtype=tf.float32, shape=filter_shape,
                                       initializer=self.filter_initializer, regularizer=self.filter_regularizer)
        
        if self.use_bias:
            shape = [1, 1, 1, 1, self.out_height, self.channels]
            self.bias = self.add_weight(name=self.name+'_b', dtype=tf.float32, shape=shape,
                                        initializer=self.bias_initializer, regularizer=self.bias_regularizer)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs the 3D G-Convolution on the input, where weights are only shared horizontally.

        Args:
            inputs (tf.Tensor): The input of shape [batch_size, width, depth, in_transformations, height, in_channels].

        Returns:
            tf.Tensor: The output of shape [batch_size, out_width, out_depth, out_transformations, out_height, out_channels].
        """
        with tf.name_scope(self.name) as scope:
            batch_size = tf.shape(inputs)[0] # batch size is unknown during construction, thus use tf.shape
            
            padded_filters = self.pad_filters(self.filters) # converts 3d kernels to 2d kernels over full height
        
            padded_inputs = self.pad_inputs(inputs) # add conventional padding to inputs
            
            filters_reshaped = tf.reshape(padded_filters, [self.h_ksize, self.h_ksize, 
                                                           self.in_transformations*self.padded_in_height*self.in_channels, 
                                                           self.out_height*self.channels])
            inputs_reshaped = tf.reshape(padded_inputs, padded_inputs.shape[:4] + [np.prod(padded_inputs.shape[-2:])])
        
            output_reshaped = splitgconv2d.gconv2d(input=inputs_reshaped, filters=filters_reshaped, 
                                                   strides=self.strides[:2], padding='VALID', gconv_indices=self.gconv_indices, gconv_shape_info=self.gconv_shape_info, name=self.name)
            
            output = tf.reshape(output_reshaped, output_reshaped.shape[:4] + [self.out_height, self.channels])
            
            if self.use_bias:
                output = tf.add(output, self.bias)
                
            return output
        
    def pad_filters(self, filters: tf.Tensor) -> tf.Tensor:
        """Pads the 3D convolution filters of height z with zeros to the height of the input data.

        Args:
            filters (tf.Tensor): The filter of shape [x, y, in_transformations, z, in_channels, out_height, out_channels].

        Returns:
            tf.Tensor: The padded filter of shape [x, y, in_transformations, in_height, in_channels, out_height, out_channels].
        """
        
        padding_shape = filters.shape[:3] + [self.padded_in_height-self.v_ksize] + filters.shape[4:]
        padding = tf.zeros(padding_shape)
        
        # add padding below the filters
        padded_filters = tf.concat([filters, padding], axis=3)
        
        # roll the padded filter so that the 3D filters are at their respective heights
        padded_filters = tf.transpose(padded_filters, [5, 0, 1, 2, 3, 4, 6]) # -> (out_height,x,y,t_in,h,c_in,c_out)
        rolled_filters = tf.map_fn(
            lambda x: tf.roll(x[0], shift=x[1]*self.strides[2], axis=3), # roll along the h axis of the filter
            elems=[padded_filters, tf.range(0, self.out_height)],
            fn_output_signature=padded_filters.dtype
        )
        rolled_filters = tf.transpose(rolled_filters, [1, 2, 3, 4, 5, 0, 6]) # -> (x,y,t_in,h,c_in,out_height,c_out)
        
        return rolled_filters
    
    def pad_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        """Adds padding to the input data of the layer.

        Args:
            inputs (tf.Tensor): Must be of shape [batch_size, width, depth, in_transformations, height, in_channels]

        Returns:
            tf.Tensor: The padded input of shape [batch_size, width', depth', in_transformations, height, in_channels]
        """
        if self.h_padding != 'VALID':        
            width_padding = required_padding(self.h_ksize, self.in_width, self.strides[0])
            depth_padding = required_padding(self.h_ksize, self.in_depth, self.strides[1])
            # batch, width, depth, transformation, height, channel
            padding = [[0, 0], width_padding, depth_padding, [0, 0], [0, 0], [0, 0]] 
            inputs = tf.pad(inputs, padding, mode='CONSTANT' if self.h_padding == 'SAME' else self.h_padding)

        if self.v_padding != 'VALID':
            height_padding = required_padding(self.v_ksize, self.in_height, self.strides[2])
            # batch, width, depth, transformation, height, channel
            padding = [[0, 0], [0, 0], [0, 0], [0, 0], height_padding, [0, 0]]
            inputs = tf.pad(inputs, padding, mode='CONSTANT' if self.v_padding == 'SAME' else self.v_padding)
            
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