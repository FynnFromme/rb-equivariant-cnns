"""
Implementation of SE(2,N) equivariant convolutions.
Details in MICCAI 2018 paper: "Roto-Translation Covariant Convolutional Networks for Medical Image Analysis".

This module is a modified version of an earlier work by Daniel Worrall under the licence below.

See https://github.com/danielewworrall/harmonicConvolutions for the original implementation.

Modifications include:
- Documentation
- Refactoring
- Tensorflow 2 migration
- Bug fixes
- Unified interface with other models
________________________________________________________________________

Released in June 2018
@author: EJ Bekkers, Eindhoven University of Technology, The Netherlands
@author: MW Lafarge, Eindhoven University of Technology, The Netherlands
________________________________________________________________________

Copyright 2018 Erik J Bekkers and Maxime W Lafarge, Eindhoven University 
of Technology, the Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
________________________________________________________________________
"""

import numpy as np
import tensorflow as tf
from . import rotation_matrix

# THE CONVOLUTION LAYERS

def z2_se2n(input_tensor: tf.Tensor, filter: tf.Variable, orientations: int, periodicity: float = 2*np.pi,
            diskMask: bool = True, strides: tuple = (1,1), padding: str = 'VALID', name: str = 'z2_se2n_conv'
            ) -> tuple[tf.Tensor, tf.Tensor]:
    """Performs a group convolution lifting from Z2 to SE(2,N).

    Args:
        input_tensor (tf.Tensor): Z2 input tensor with shape: [batch_size, h, w, in_channels].
        filter (tf.Variable): The filter bank of shape [filter_size, filter_size, in_channels, out_channels].
        orientations (int): Specifying the number of rotations the filter is applied at.
        periodicity (float, optional): The range of rotations. Defaults to 2*np.pi.
        diskMask (bool, optional): Whether values outside a circular mask are set to zero. Defaults to True.
        strides (tuple, optional): The stride used in the conv operation. Defaults to (1,1).
        padding (str, optional): Padding of the conv operation ('VALID' or 'SAME'). Defaults to 'VALID'.
        name (str, optional): The name of the tensorflow operation. Defaults to 'z2_se2n_conv'.

    Returns:
        tf.Tensor: The resulting SE(2,N) feature maps of shape [batch_size, h', w', orientations, out_channels].
        tf.Tensor: Theformated filters, i.e., the full stack of rotated filters of shape
            [orientations, filter_size, filter_size, in_channels, out_channels]
    """
    batch_size = tf.shape(input_tensor)[0] # batch size is unknown during construction, thus use tf.shape
    
    # filter dimensions
    filter_size_h, filter_size_w, channels_in, channels_out = map(int, filter.shape)

    # Precompute a rotated stack of filters, shape=[orientations,h,w,in_channels,out_channels]
    filter_stack = rotate_lifting_filters(filter, orientations, periodicity=periodicity, diskMask=diskMask)

    # Format the filter stack as a 2D filter stack, shape=[h,w,in_channels,orientations*out_channels]
    filters_as_if_2D = tf.transpose(filter_stack, [1, 2, 3, 0, 4])
    filters_as_if_2D = tf.reshape(filters_as_if_2D, [filter_size_h, filter_size_w, channels_in, orientations * channels_out])

    # Perform the 2D convolution
    layer_output = tf.nn.conv2d(
        input=input_tensor,
        filters=filters_as_if_2D,
        strides=strides,
        padding=padding,
        name=name)

    # Reshape to an SE2 image (split the orientation and out_channels axis)
    output_shape_3d = tf.concat([[batch_size], layer_output.shape[1:3], [orientations, channels_out]], axis=0)
    layer_output = tf.reshape(layer_output, output_shape_3d)

    return layer_output, filter_stack


def se2n_se2n(input_tensor: tf.Tensor, filter: tf.Variable, periodicity: float = 2*np.pi, diskMask: bool = True, 
              strides: tuple = (1,1), padding: str = 'VALID', name: str = 'se2n_se2n_conv') -> tuple[tf.Tensor, tf.Tensor]:
    """Performs a group convolution from SE(2,N) to SE(2,N).

    Args:
        input_tensor (tf.Tensor): SE(2,N) input tensor with shape: [batch_size, orientations, h, w, in_channels].
        filter (tf.Variable): The filter bank of shape [filter_size, filter_size, orientations, in_channels, out_channels].
        periodicity (float, optional): The range of rotations. Defaults to 2*np.pi.
        diskMask (bool, optional): Whether values outside a circular mask are set to zero. Defaults to True.
        strides (tuple, optional): The stride used in the conv operation. Defaults to (1,1).
        padding (str, optional): Padding of the conv operation ('VALID' or 'SAME'). Defaults to 'VALID'.
        name (str, optional): The name of the tensorflow operation. Defaults to 'se2n_se2n_conv'.

    Returns:
        tf.Tensor: The resulting SE(2,N) feature maps of shape [batch_size, h', w', orientations, out_channels].
        tf.Tensor: Theformated filters, i.e., the full stack of rotated filters of shape
            [orientations, filter_size, filter_size, orientations, in_channels, out_channels]
    """
    
    batch_size = tf.shape(input_tensor)[0] # batch size is unknown during construction, thus use tf.shape

    # filter dimensions
    filter_size_h, filter_size_w, orientations, channels_in, channels_out = map(int, filter.shape)

    # Preparation for group convolutions
    # Precompute a rotated stack of se2 filters
    # With shape: [orientations, h, w, orientations, channels_in, channels_out]
    filter_stack = rotate_gconv_filters(filter, periodicity, diskMask)

    # Group convolutions are done by integrating over [x,y,theta,channels_in] for each translation and 
    # rotation of the filter
    # We compute this integral by doing standard 2D convolutions (translation part) for each rotated 
    # version of the filter (rotation part)
    # In order to efficiently do this we use 2D convolutions where the theta and input-channel axes are merged 
    # (thus treating the SE2 image as a 2D feature map)

    # Prepare the input tensor (merge the orientation and channel axis) for the 2D convolutions:
    input_shape_2d = tf.concat([[batch_size], input_tensor.shape[1:3], [orientations*channels_in]], axis=0)
    input_tensor_as_if_2D = tf.reshape(input_tensor, input_shape_2d)
                                       

    # Reshape the filters for 2D convolutions (orientation+channels_in axis are merged, 
    # rotation+channels_out axis are merged)
    filters_as_if_2D = tf.transpose(filter_stack, [1, 2, 3, 4, 0, 5])
    filters_as_if_2D = tf.reshape(
        filters_as_if_2D, [filter_size_h, filter_size_w, orientations * channels_in, orientations * channels_out])

    # Perform the 2D convolutions
    layer_output = tf.nn.conv2d(
        input=input_tensor_as_if_2D,
        filters=filters_as_if_2D,
        strides=strides,
        padding=padding,
        name=name)

    # Reshape into an SE2 image (split the rotation and channels_out axis)
    output_shape_3d = tf.concat([[batch_size], layer_output.shape[1:3], [orientations, channels_out]], axis=0)
    layer_output = tf.reshape(layer_output, output_shape_3d)

    return layer_output, filter_stack


# FILTER ROTATION FUNCTIONS

def rotate_lifting_filters(filter: tf.Tensor, orientations: int, periodicity: float = 2*np.pi, 
                           diskMask: bool = True) -> tf.Tensor:
    """Rotates the set of 2D lifting filters. Filter must have odd size.

    Args:
        filter (tf.Tensor): The filter bank of shape [filter_size, filter_size, in_channels, out_channels].
        orientations (int): Specifying the number of rotations the filter is applied at.
        periodicity (float, optional): The range of rotations. Defaults to 2*np.pi.
        diskMask (bool, optional): Whether values outside a circular mask are set to zero. Defaults to True.

    Returns:
        tf.Tensor: The set of rotated filters of shape [orientations, filter_size, filter_size, in_channels, out_channels].
    """

    # Unpack the shape of the input filter
    filter_size_h, filter_size_w, channels_in, channels_out = map(int, filter.shape)

    # Flatten the baseline filter
    # Resulting shape: [h*w, channels_in*channels_out]
    filter_flat = tf.reshape(filter, [filter_size_h * filter_size_w, channels_in * channels_out])

    # Generate a set of rotated filters via rotation matrix multiplication
    # Resulting shape: [orientations*h*w, h*w]
    rotOp_matrix = rotation_matrix.MultiRotationOperatorMatrix(
        [filter_size_h, filter_size_w],
        orientations,
        periodicity=periodicity,
        diskMask=diskMask)
    
    rotOp_matrix = tf.cast(rotOp_matrix, dtype=tf.float32)

    # Matrix multiplication
    # Resulting shape: [orientations*h*w, channels_in*channels_out]
    set_of_rotated_filters = tf.matmul(rotOp_matrix, filter_flat)

    # Reshaping
    # Resulting shape: [orientations, h, w, channels_in,channels_out]
    set_of_rotated_filters = tf.reshape(
        set_of_rotated_filters, [orientations, filter_size_h, filter_size_w, channels_in, channels_out])

    return set_of_rotated_filters


def rotate_gconv_filters(filter, periodicity=2 * np.pi, diskMask=True):
    """Rotates the set of SE2 filters. Rotation of SE2 filters involves planar rotations and a shift in orientation.
    Filter must have odd size.

    Args:
        filter (tf.Tensor): The filter bank of shape [filter_size, filter_size, orientations, in_channels, out_channels].
        periodicity (float, optional): The range of rotations. Defaults to 2*np.pi.
        diskMask (bool, optional): Whether values outside a circular mask are set to zero. Defaults to True.

    Returns:
        tf.Tensor: The set of rotated filters of shape 
            [orientations, filter_size, filter_size, orientations, in_channels, out_channels].
    """
    # Rotation of an SE2 filter consists of two parts:
    # PART 1. Planar rotation
    # PART 2. A shift in theta direction

    # Unpack the shape of the input filter
    filter_size_h, filter_size_w, orientations, channels_in, channels_out = map(int, filter.shape)

    # PART 1 (planar rotation)
    # Flatten the baseline filter
    # Resulting shape: [h*w,orientations*channels_in*channels_out]
    #
    filter_flat = tf.reshape(filter, [filter_size_h * filter_size_w, orientations * channels_in * channels_out])

    # Generate a set of rotated filters via rotation matrix multiplication
    # Resulting shape: [orientations*h*w,h*w]
    rotOp_matrix = rotation_matrix.MultiRotationOperatorMatrix(
        [filter_size_h, filter_size_w],
        orientations,
        periodicity=periodicity,
        diskMask=diskMask)
    
    rotOp_matrix = tf.cast(rotOp_matrix, dtype=tf.float32)

    # Matrix multiplication (each 2D plane is now rotated)
    # Resulting shape: [orientations*h*w, orientations*channels_in*channels_out]
    filters_planar_rotated = tf.matmul(rotOp_matrix, filter_flat)
    filters_planar_rotated = tf.reshape(
        filters_planar_rotated, [orientations, filter_size_h, filter_size_w, orientations, channels_in, channels_out])

    # PART 2 (shift in theta direction)
    set_of_rotated_filters = [None] * orientations
    for orientation in range(orientations):
        # [h,w,orientations,channels_in,channels_out]
        filters_temp = filters_planar_rotated[orientation]
        # [h,w,channels_in,channels_out,orientations]
        filters_temp = tf.transpose(filters_temp, [0, 1, 3, 4, 2])
        # [h*w*channels_in*channels_out*orientations]
        filters_temp = tf.reshape(filters_temp, [filter_size_h * filter_size_w * channels_in * channels_out, orientations])
        # Roll along the orientation axis
        roll_matrix = tf.constant(np.roll(np.identity(orientations), orientation, axis=1), dtype=filters_temp.dtype)
        filters_temp = tf.matmul(filters_temp, roll_matrix)
        filters_temp = tf.reshape(
            filters_temp, [filter_size_h, filter_size_w, channels_in, channels_out, orientations])  # [Nx,Ny,Nin,Nout,Ntheta]
        filters_temp = tf.transpose(filters_temp, [0, 1, 4, 2, 3])
        set_of_rotated_filters[orientation] = filters_temp

    return tf.stack(set_of_rotated_filters)