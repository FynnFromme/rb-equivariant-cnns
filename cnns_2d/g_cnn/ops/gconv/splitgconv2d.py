
import numpy as np
import tensorflow as tf

from .make_gconv_indices import (flatten_indices, make_c4_p4_indices,
                                 make_c4_z2_indices, make_d4_p4m_indices,
                                 make_d4_z2_indices)
from .transform_filter import transform_filter_2d_nhwc


def gconv2d(input: tf.Tensor, filters: tf.Variable, strides: list, padding: str, gconv_indices: np.ndarray, 
            gconv_shape_info: tuple, data_format: str = 'NHWC', name: str = None) -> tf.Tensor:
    """Tensorflow implementation of the group convolution.
    This function has the same interface as the standard convolution nn.conv2d, except for two new parameters,
    gconv_indices and gconv_shape_info. These can be obtained from gconv2d_util(), and are described below.

    Args:
        input (tf.Tensor): The input of shape [batches, height, width, in_transformations, in_channels].
        filter (tf.Variable): The learned filter bank of shape [ksize, ksize, in_transformations*in_channels, out_channels].
            The shape for filter can be obtained from `gconv2d_util()`
        strides (list): A list of 4 integers. Index 1 and 2 refer to the spatial domain.
        padding (str): A string from: "SAME", "VALID". The type of padding algorithm to use.
        gconv_indices (np.ndarray): indices used in the filter transformation step of the G-Conv.
            Can be obtained from `gconv2d_util` or using a command like `flatten_indices(make_c4_z2_indices(ksize=3))`.
        gconv_shape_info (tuple): a tuple containing 
            (num output channels, num output transformations, num input channels, num input transformations, kernel size)
            Can be obtained from `gconv2d_util`
        data_format (str, optional): The order of axes. Currently only NHWC is supported. Defaults to 'NHWC'.
        name (str, optional): The name used for the tensorflow operation. Defaults to None.

    Raises:
        NotImplemented: If the data_format is not supported.

    Returns:
        tf.Tensor: The convolution output as a tensor with (batch, height, width, out_transformations, out_channels) axes.
    """

    if data_format != 'NHWC':
        raise NotImplemented('Currently only NHWC data_format is supported. Got:' + str(data_format))
    
    batch_size = tf.shape(input)[0] # batch size is unknown during construction, thus use tf.shape

    # Transform the filters
    transformed_filter = transform_filter_2d_nhwc(w=filters, flat_indices=gconv_indices, shape_info=gconv_shape_info)

    # Transform input
    no, nto, ni, nti, n = gconv_shape_info
    input = tf.reshape(input, tf.concat([[batch_size], input.shape[1:-2], [nti*ni]], axis=0))

    # Convolve input with transformed filters
    conv = tf.nn.conv2d(input=input, filters=transformed_filter, strides=strides, padding=padding,
                        data_format=data_format, name=name)
    
    # Transform output back
    conv = tf.reshape(conv, tf.concat([[batch_size], conv.shape[1:-1], [nto, no]], axis=0))

    return conv


def gconv2d_util(h_input: str, h_output: str, in_channels: int, out_channels: int, ksize: int) -> tuple:
    """Convenience function for setting up static data required for the G-Conv.
     This function returns:
      1) an array of indices used in the filter transformation step of gconv2d
      2) shape information required by gconv2d
      3) the shape of the filter tensor to be allocated and passed to gconv2d

    Args:
        h_input (str): One of ('Z2', 'C4', 'D4'). Use 'Z2' for the first layer. Use 'C4' or 'D4' for later layers.
        h_output (str): One of ('C4', 'D4'). What kind of transformations to use (rotations or roto-reflections).
        in_channels (int): The number of input channels. Note: this refers to the number of (3D) channels on the group.
            The number of 2D channels will be 1, 4, or 8 times larger, depending the value of h_input.
        out_channels (int): The number of output channels. Note: this refers to the number of (3D) channels on the group.
            The number of 2D channels will be 1, 4, or 8 times larger, depending on the value of h_output.
        ksize (int): The spatial size of the filter kernels (typically 3, 5, or 7).

    Raises:
        ValueError: If the pair of input and output domain is not supported.

    Returns:
        1) gconv_indices: The index mapping for filter rotations of the form [filter_rot,input_rot,upos,vpos] -> index.
        2) gconv_shape_info: Shape information for gconv2d (out_channels,out_rotations,in_channels,in_rotations,ksize).
        3) w_shape: Size of the learned filter bank (ksize,ksize,in_rotations*in_channels,out_channels).
    """

    # index mapping for output transformations (rotations or rotationflips)
    if h_input == 'Z2' and h_output == 'C4':
        gconv_indices = flatten_indices(make_c4_z2_indices(ksize=ksize))
        nti = 1
        nto = 4
    elif h_input == 'C4' and h_output == 'C4':
        gconv_indices = flatten_indices(make_c4_p4_indices(ksize=ksize))
        nti = 4
        nto = 4
    elif h_input == 'Z2' and h_output == 'D4':
        gconv_indices = flatten_indices(make_d4_z2_indices(ksize=ksize))
        nti = 1
        nto = 8
    elif h_input == 'D4' and h_output == 'D4':
        gconv_indices = flatten_indices(make_d4_p4m_indices(ksize=ksize))
        nti = 8
        nto = 8
    else:
        raise ValueError('Unknown (h_input, h_output) pair:' + str((h_input, h_output)))

    w_shape = (ksize, ksize, nti*in_channels, out_channels)
    gconv_shape_info = (out_channels, nto, in_channels, nti, ksize)
    return gconv_indices, gconv_shape_info, w_shape