import tensorflow as tf
from tensorflow import keras

class RB3DConv(keras.Layer):
    def __init__(self, ksize: int | tuple, channels: int, *args, **kwargs):
        """A 3D Rayleigh-Bénard Convolutional layer convolves the input with a 3d filter, while parameters
        are only shared horizontally since 3D Rayleigh-Bénard is not translation equivariant in the vertical 
        direction.
        
        Both input and output should have the following shape: [batch_size, width, depth, height, channels].
        
        Currently only VALID padding is supported.

        Args:
            ksize (int | tuple): The size of the filter. If ksize is an integer, the size is used in all directions.
            channels (int): The number of output channels.
            *args, **kwargs: Please refer to the keras documentation for additional arguments.
        """
        super().__init__(*args, **kwargs)
        self.ksize = (ksize, ksize, ksize) if type(ksize) is int else ksize
        self.channels = channels
        
    def build(self, input_shape: list):
        """Initializes the filters of the layer.

        Args:
            input_shape (list): The input as shape [batch_size, width, depth, height, channels]
        """
        self.batch_size, self.in_width, self.in_depth, self.in_height, self.in_channels = input_shape
        self.out_height = self.in_height-(self.ksize[2]-1)
        
        filter_shape = [*self.ksize, self.in_channels, self.out_height, self.channels]
        
        self.filters = self.add_weight(name=self.name+'_w', dtype=tf.float32, shape=filter_shape)
    
    def call(self, input: tf.Tensor) -> tf.Tensor:
        """Convolves the input with a 3D filter, where weights are only shared horizontally.

        Args:
            input (tf.Tensor): The input to the layer of shape [batch_size, width, depth, height, channels].

        Returns:
            tf.Tensor: The output of the layer of shape [batch_size, out_width, out_depth, out_height, out_channels].
        """
        
        padded_filters = self.pad_filters(self.filters)
        
        input_reshaped = tf.reshape(input, [self.batch_size, 
                                            self.in_width, self.in_depth, 
                                            self.in_height*self.in_channels])
        filters_reshaped = tf.reshape(padded_filters, [self.ksize[0], self.ksize[1], 
                                                       self.in_height*self.in_channels,
                                                       self.out_height*self.channels])
        
        # conv2d output has shape [batch_size, out_width, out_depth, out_height*out_channels]
        output_reshaped = tf.nn.conv2d(input_reshaped, filters_reshaped, strides=1, padding='VALID')
        
        output = tf.reshape(output_reshaped, output_reshaped.shape[:3] + [self.out_height, self.channels])
        
        return output
        
    def pad_filters(self, filters: tf.Tensor) -> tf.Tensor:
        """Pads the 3D convolution filters of height z with zeros to the height of the input data.

        Args:
            filters (tf.Tensor): The filter of shape [x, y, z, in_channels, out_height, out_channels].

        Returns:
            tf.Tensor: The padded filter of shape [x, y, in_height, in_channels, out_height, out_channels].
        """
        
        padding_shape = filters.shape[:2] + [self.in_height-self.ksize[2]] + filters.shape[3:]
        padding = tf.zeros(padding_shape)
        
        # add padding below the filters
        padded_filters = tf.concat([filters, padding], axis=2)
        
        # roll the padded filter so that the 3D filters are at their respective heights
        padded_filters = tf.transpose(padded_filters, [4, 0, 1, 2, 3, 5]) # -> (out_height,x,y,h,c_in,c_out)
        rolled_filters = tf.map_fn(
            lambda x: tf.roll(x[0], shift=x[1], axis=2), # roll along the h axis of the filter
            elems=[padded_filters, tf.range(0, self.out_height)],
            fn_output_signature=padded_filters.dtype
        )
        rolled_filters = tf.transpose(rolled_filters, [1, 2, 3, 4, 0, 5]) # -> (x,y,z,c_in,out_height,c_out)
        
        return rolled_filters
    