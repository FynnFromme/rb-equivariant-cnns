"""
Core Harmonic Convolution Implementation
"""

from typing import Callable

import numpy as np
import tensorflow as tf


def h_conv(x: tf.Tensor, w: dict[int, tuple[tf.Variable, tf.Variable]], strides: tuple = (1,1),
           padding: str = 'VALID', max_order: int = 1, conjugate_weights: bool = False, name: str = 'h_conv'
           ) -> tf.Tensor:
    """Inter-order (cross-stream) convolutions can be implemented as single convolution. For this we store 
    data as 6D tensors and filters as 8D tensors, at convolution, we reshape down to 4D tensors and expand again.

    Args:
        x (tf.Tensor): The feature maps of shape [batch_size, h, w, in_orders, in_complex, in_channels]
        w (dict[int, tuple[tf.Variable, tf.Variable]]): Dictionary containing the real and imaginary parts of the 
            grid filters for the different rotation orders. Real and imaginary parts have both the shape 
            [k, k, in_channels, out_channels].
        strides (tuple, optional): The stride used in the conv operation. Defaults to (1,1).
        padding (str, optional): Padding of the conv operation ('VALID' or 'SAME'). Defaults to 'VALID'.
        max_order (int, optional): The output rotation orders are 0...max_order. Defaults to 1.
        conjugate_weights (bool, optional): Whether to use conjugate weights for negative orders. Defaults to False.
        name (str, optional): The name of the tensorflow operation. Defaults to 'h_conv'.

    Returns:
        tf.Tensor: The output feature maps of shape [batch_size, h', w', out_orders, out_complex, out_channels].
    """
    in_range = (0, x.shape[3]-1)
    out_range = (0, max_order)
    return h_range_conv(x, w, strides, padding, in_range, out_range, conjugate_weights, name)


def h_range_conv(x: tf.Tensor, w: dict[int, tuple[tf.Variable, tf.Variable]], strides: tuple = (1,1), 
                 padding: str = 'VALID', in_range: tuple = (0,1), out_range: tuple = (0,1), 
                 conjugate_weights: bool = False, name: str = 'h_conv') -> tf.Tensor:
    """Inter-order (cross-stream) convolutions can be implemented as single convolution. For this we store data as 6D 
    tensors and filters as 8D tensors, at convolution, we reshape down to 4D tensors and expand again.
    
    In contrast to `h_range_conv`, the rotation orders don't start at 0 and can be in any range.

    Args:
        x (tf.Tensor): The feature maps of shape [batch_size, h, w, in_orders, complex, in_channels]
        w (dict[int, tuple[tf.Variable, tf.Variable]]): Dictionary containing the real and imaginary parts of the 
            grid filters for the different rotation orders. Real and imaginary parts have both the shape 
            [k, k, in_channels, out_channels]
        strides (tuple, optional): The stride used in the conv operation. Defaults to (1,1).
        padding (str, optional): Padding of the conv operation ('VALID' or 'SAME'). Defaults to 'VALID'.
        in_range (tuple, optional): The range of rotation orders of the input feature maps as a 2 tuple. Including both
            start and end. Defaults to (0,1).
        out_range (tuple, optional): The range of rotation orders of the output feature maps as a 2 tuple. Including both
            start and end. Defaults to (0,1).
        conjugate_weights (bool, optional): Whether to use conjugate weights for negative orders. Defaults to False.
        name (str, optional): The name of the tensorflow operation. Defaults to 'h_conv'.

    Returns:
        tf.Tensor: The output feature maps of shape [batch_size, h', w', out_orders, out_complex, out_channels].
    """
    xsh = list(x.shape) # [batch_size,h,w,in_orders,in_complex,in_channels]
    batch_size = tf.shape(x)[0] # batch size is unknown during construction, thus use tf.shape

    # reshaping input to a stack of 2D feature maps [batch_size,h,w,in_orders*in_complex*in_channels]
    x_ = tf.reshape(x, tf.concat([[batch_size], xsh[1:3], [-1]], axis=0))

    # The script below constructs the stream-convolutions as one big filter w_. For each output order, 
    # run through each input order and copy-paste the filter for that convolution.

    # w_ contains the filters for all combinations of output and input order. During construction it's 
    # a list of out_complex*out_order many tensors with shape [k,k,in_orders*in_complex*in_channels,out_channels].
    # Later it is concatinated to a single tensor.
    w_ = []
    for output_order in range(out_range[0], out_range[1]+1):
        # wr and wi are the filters resulting in the current output_order. The application of wr results in 
        # the real response and wi in the imaginary response.
        # both are a list of in_complex*in_orders many tensors of shape [k,k,in_channels,out_channels]
        # The first element of the in_orders many pairs is convolved with Re(x) and the second element 
        # with Im(x), each on the respective in_order
        wr = []
        wi = []
        for input_order in range(in_range[0], in_range[1]+1):
            # negative orders use the conjugate weights
            weight_order = output_order - input_order # rotation order of filter
            if conjugate_weights:
                # for negative orders use conjugate of positive order
                weights = w[np.abs(weight_order)] # shape=[complex,k,k,in_channels,out_channels]
                sign = np.sign(weight_order)
            else:
                weights = w[weight_order] # shape=[complex,k,k,in_channels,out_channels]
                sign = 1
            
            # Complex Convolution: W*x = Re(W)*Re(x)-Im(W)*Im(x) + i(Im(W)*Re(x)+Re(W)*Im(x))
            if xsh[4] == 2: # for complex feature map
                wr += [weights[0],-sign*weights[1]]
                wi += [sign*weights[1],weights[0]]
            else: # for real feature map
                wr += [weights[0]]
                wi += [sign*weights[1]]
        
        # wr and wi are respectively concatinated to one tensor of shape 
        # [k,k,in_orders*input_complex*in_channels,out_channels] 
        w_ += [tf.concat(wr, axis=2), tf.concat(wi, axis=2)] 
        
    # w_ is concatinated to shape [k,k,in_orders*in_complex*in_channels,out_order*out_complex*out_channels]
    w_ = tf.concat(w_, axis=3) 

    # Convolve
    # x_ of shape [batch_size,h,w,in_orders*in_complex*in_channels]
    # with w_ of shape [k,k,in_orders*in_complex*in_channels,out_order*out_complex*out_channels]
    # resulting in Y of shape [batch_size,h',w',out_order*out_complex*out_channels]
    y = tf.nn.conv2d(x_, filters=w_, strides=strides, padding=padding, name=name)
    
    # Reshape result into appropriate shape [batch_size,h',w',out_orders,out_complex,out_channels]
    ysh = y.shape
    orders = out_range[1] - out_range[0] + 1
    new_shape = tf.concat([[batch_size], ysh[1:3],[orders,2],[int(ysh[3]/(2*(orders)))]], axis=0)
    return tf.reshape(y, new_shape)


##### NONLINEARITIES #####
def h_nonlin(x: tf.Tensor, fnc: Callable, b: tf.Variable, eps: float = 1e-4) -> tf.Tensor:
    """Apply the nonlinearity described by the function handle fnc: R -> R+ to
    the magnitude of x after adding a bias. CAVEAT: fnc must map to the non-negative reals R+.

    Args:
        x (tf.Tensor): Feature maps of shape [batch_size, h, w, orders, complex, channels].
        fnc (Callable): Function handle for a nonlinearity. MUST map to non-negative reals R+.
        b (tf.Variable): Bias that will be added to the magnitudes before applying the nonlinearity.
            Must have shape [1, 1, 1, orders, 1, channels].
        eps (float, optional): Regularization since gradient of magnitudes is infinite at zero. Defaults to 1e-4.

    Returns:
        tf.Tensor: Feature maps with the nonlinearity applied to the magnitudes of x.
            Has the same shape as the input.
    """
    magnitudes = stack_magnitudes(x, eps, keepdims=True)
    
    # add bias to nonlinearty
    new_magnitudes = fnc(tf.add(magnitudes, b))
    
    # apply nonlinearity to magnitudes and update magnitudes of x
    return update_magnitudes(x, new_magnitudes, magnitudes)


def mean_pooling(x: tf.Tensor, ksize: tuple = (2,2), strides: tuple = (2,2), padding: str = 'VALID', 
                 name: str = 'mean_pooling') -> tf.Tensor:
    """Performs mean pooling on complex-valued feature maps. The complex mean
    on a local receptive field, is performed as mean(real) + i*mean(imag)

    Args:
        x (tf.Tensor): Feature maps of shape [batch_size, h, w, orders, complex, channels].
        ksize (tuple, optional): The pooling region size as a 2-tuple. Defaults to (2,2).
        strides (tuple, optional): The stride size as a 2 tuple. Defaults to (2,2).
        padding (str, optional): Padding of the pool operation ('VALID' or 'SAME'). Defaults to 'VALID'.
        name (str, optional): The name of the tensorflow operation. Defaults to 'h_conv'.

    Returns:
        tf.Tensor: The pooled data of shape [batch_size, h', w', orders, complex, channels].
    """
    xsh = x.shape # [batch_size,h,w,orders,complex,channels]
    batch_size = tf.shape(x)[0] # batch size is unknown during construction, thus use tf.shape
    
    # collapse the order, complex, and channel dimensions, # shape=[batch_size,h,w,orders*complex*channels]
    x_ = tf.reshape(x, tf.concat([[batch_size], xsh[1:3],[-1]], axis=00))
    
    # perform pooling
    y = tf.nn.avg_pool2d(input=x_, ksize=ksize, strides=strides, padding=padding, name=name)
    
    # reshape back
    ysh = y.shape
    new_shape = tf.concat([[batch_size], ysh[1:3],xsh[3:]], axis=0) # shape=[batch_size,h',w',orders,complex,channels]
    
    return tf.reshape(y, new_shape)


def update_magnitudes(x: tf.Tensor, new_magnitudes: tf.Tensor, old_magnitudes: tf.Tensor) -> tf.Tensor:
    """Updates the magnitudes of a data tensor of shape [batch_size, h, w, orders, complex, channels].

    Args:
        x (tf.Tensor): The input data tensor.
        new_magnitudes (tf.Tensor): The new magnitudes of shape [batch_size, h, w, orders, 1, channels].
        old_magnitudes (tf.Tensor): The old magnitudes of shape [batch_size, h, w, orders, 1, channels].

    Returns:
        tf.Tensor: The data with updated magnitudes. Has the same shape as the input.
    """
    return x * tf.math.divide(new_magnitudes, old_magnitudes)


def sum_magnitudes(x: tf.Tensor, eps: float = 1e-4, keepdims: bool = True) -> tf.Tensor:
    """Taking the magnitudes of the complex feature maps and summing them over the channels.

    Args:
        x (tf.Tensor): Feature map tensor of shape [batch_size, h, w, orders, complex, channels].
        eps (float, optional): Regularization since gradient of magnitudes is infinite at zero. Defaults to 1e-4.
        keepdims (bool, optional): Whether to keep (True) or collapse (False) summed dimensions. 
            Defaults to True.

    Returns:
        tf.Tensor: The magnitudes within the featuremaps as a [batch_size, h, w, orders, 1, 1] tensor.
            If `keepdims=False` the shape is [batch_size, h, w, orders]
    """
    magnitudes = stack_magnitudes(x, eps, keepdims)
    
    channels_axis = 5 if keepdims else 4
    return tf.reduce_sum(magnitudes, axis=[channels_axis], keepdims=keepdims)


def stack_magnitudes(x: tf.Tensor, eps: float = 1e-4, keepdims: bool = True) -> tf.Tensor:
    """Taking the magnitudes of the complex feature maps.

    Args:
        x (tf.Tensor): Feature map tensor of shape [batch_size, h, w, orders, complex, channels].
        eps (float, optional): Regularization since gradient of magnitudes is infinite at zero. Defaults to 1e-4.
        keepdims (bool, optional): Whether to keep (True) or collapse (False) complex dimension. 
            Defaults to True.

    Returns:
        tf.Tensor: The magnitudes within the featuremaps as a [batch_size, h, w, orders, 1, channels] tensor.
            If `keepdims=False` the shape is [batch_size, h, w, orders, channels]
    """
    # Computing the magnitudes of feature map
    r = tf.reduce_sum(tf.square(x), axis=[4], keepdims=keepdims) # Re^2+Im^2
    return tf.sqrt(tf.maximum(r,eps))
