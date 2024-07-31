import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

import sys, os
tests_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(tests_dir, '..'))

import h_net.layers


def get_model(in_shape: tuple, *layers: keras.Layer) -> keras.Model:
    """Creates a tensorflow model with the given input shape and se2n_cnn.layers.

    Args:
        in_shape (tuple): The shape of the input.
        *layers (keras.Layer): The keras layers of the model.

    Returns:
        keras.Model: The resulting model.
    """
    return keras.Sequential([
            keras.layers.InputLayer(shape=in_shape, batch_size=1),
            *layers
        ])


def check_real_equivariance(model: keras.Model, in_shape: tuple, in_rot_axes: tuple | None = (1,2),
                            out_rot_axes: tuple | None = (1,2)) -> bool:
    """Checks the 90° rotational equivariance of the model.
    
    The output must either have a complex dimension of size one or no complex dimension at all

    Args:
        model (keras.Model): The model to test equivariance of.
        in_shape (tuple): The shape of the input (excluding batch dimension).
        in_rot_axes (tuple | None, optional): The axes of the input to rotate or None if input not rotated. Defaults to (1,2).
        out_rot_axes (tuple | None, optional): The axes of the output to rotated or None if output not rotated. Defaults to (1,2).

    Returns:
        bool: Whether the model is equivariant under 90° rotations.
    """
    inpt = np.random.randn(1,*in_shape)
    inpt_rot = np.rot90(inpt, k=1, axes=in_rot_axes)
    
    output = model(inpt, training=False)
    output_rot = model(inpt_rot, training=False)
    
    output_rot_reversed = output_rot
    if out_rot_axes is not None:
        output_rot_reversed = np.rot90(output_rot_reversed, k=-1, axes=out_rot_axes)
    
    if not np.isclose(output, output_rot_reversed, rtol=1e-5, atol=1e-3).all():
        print(output)
        print(output_rot_reversed)
        print(np.isclose(output, output_rot_reversed, rtol=1e-5, atol=1e-3))
        print(np.isclose(output, output_rot_reversed, rtol=1e-3, atol=1e-3))
        print(output-output_rot_reversed)
    
    return np.isclose(output, output_rot_reversed, rtol=1e-5, atol=1e-3).all()


def check_complex_equivariance(model: keras.Model, in_shape: tuple, out_orders: tuple) -> bool:
    """Checks the 90° rotational equivariance of the model.

    Both input and output must have shape [batch,h,w,order,complex,channel]

    Args:
        model (keras.Model): The model to test equivariance of.
        in_shape (tuple): The shape of the input (excluding batch dimension).
        out_orders (tuple): The range of orders in the model output (including both start and end).

    Returns:
        bool: Whether the model is equivariant under 90° rotations.
    """
    inpt = np.random.randn(1,*in_shape)
    inpt_rot = np.rot90(inpt, k=1, axes=(1,2))
    
    output = model(inpt, training=False).numpy().transpose((0, 4, 1, 2, 3, 5))[0] # shape=[complex,h,w,order,channel]
    output_rot = model(inpt_rot, training=False).numpy().transpose((0, 4, 1, 2, 3, 5))[0] # shape=[complex,h,w,order,channel]
    
    # turn output into complex numbers
    output = output[0] + 1j*output[1] # shape=[h,w,order,channel]
    output_rot = output_rot[0] + 1j*output_rot[1] # shape=[h,w,order,channel]
    
    # check equality by comparing absolute and angle values
    abs_out = np.abs(output)
    abs_out_rot = np.abs(output_rot)
    angle_out = np.angle(output)
    angle_out_rot = np.angle(output_rot)
    
    # the angle offsets for the different rotation orders
    angle_offsets = -np.pi/2 * np.arange(out_orders[0], out_orders[1]+1)[np.newaxis, np.newaxis, :, np.newaxis]
    
    # reverse the output transformation
    abs_out_rot_reversed = np.rot90(abs_out_rot, k=-1, axes=(0,1))
    angle_out_rot_reversed = np.rot90(angle_out_rot, k=-1, axes=(0,1)) + angle_offsets
    
    # set angle to 0 if number close to 0, since angle is then not defined
    angle_out = np.where(np.isclose(abs_out, 0), 0, angle_out)
    angle_out_rot_reversed = np.where(np.isclose(abs_out_rot_reversed, 0), 0, angle_out_rot_reversed)
    
    abs_equivariant =  np.isclose(abs_out, abs_out_rot_reversed, rtol=1e-5, atol=1e-3).all()
    angle_equivariant = np.isclose(np.round(((angle_out - angle_out_rot_reversed)/np.pi),5)%2, 0, rtol=1e-5, atol=1e-3).all()

    return abs_equivariant and angle_equivariant


def check_shape(model: keras.Model, in_shape: tuple, out_shape: tuple) -> bool:
    """Checks the output shape of the model.

    Args:
        model (keras.Model): The model to check.
        in_shape (tuple): The shape of the input (excluding the batch dimension).
        out_shape (tuple): The shape of the output (excluding the batch dimension). Dimensions given as None are ignored.

    Returns:
        bool: Whether the shape of the output is as expected.
    """
    inpt = np.random.randn(1,*in_shape)
    output = model(inpt, training=False)
    
    out_shape = (1,*out_shape) # add batch dimension
    return all([output.shape[i] == out_shape[i] 
                for i, d in enumerate(out_shape)
                if d is not None])


class TestEquivariance(unittest.TestCase):
    """Checks 90° rotational equivariance as well as the output shape of every H-Net layer."""
    
    def test_HRangeConv(self):
        # Test 1: Default
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, h_net.layers.HRangeConv(channels=1, ksize=3, padding='VALID',
                                                            phase=True, in_range=(0,0), out_range=(0,1), conjugate_weights=True))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,1)))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 2, 2, 1)))
        
        # Test 2: Complex input
        in_shape = (4, 4, 1, 2, 1)
        model = get_model(in_shape, h_net.layers.HRangeConv(channels=1, ksize=3, padding='VALID',
                                                            phase=True, in_range=(0,0), out_range=(0,1), conjugate_weights=True))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,1)))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 2, 2, 1)))
        
        # Test 3: No Phase offset terms
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, h_net.layers.HRangeConv(channels=1, ksize=3, padding='VALID',
                                                            phase=False, in_range=(0,0), out_range=(0,1), conjugate_weights=True))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,1)))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 2, 2, 1)))
        
        # Test 4: Don't use conjugate weights
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, h_net.layers.HRangeConv(channels=1, ksize=3, padding='VALID',
                                                            phase=True, in_range=(0,0), out_range=(0,1), conjugate_weights=False))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,1)))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 2, 2, 1)))
        
        # Test 5: Different output rotation orders
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, h_net.layers.HRangeConv(channels=1, ksize=3, padding='VALID',
                                                            phase=True, in_range=(0,0), out_range=(0,2), conjugate_weights=True))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,2)))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 3, 2, 1)))
        
        # Test 6: Negative rotation orders
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, h_net.layers.HRangeConv(channels=1, ksize=3, padding='VALID',
                                                            phase=True, in_range=(0,0), out_range=(-2,2), conjugate_weights=True))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(-2,2)))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 5, 2, 1)))
        
        # Test 7: Negative rotation orders without conjugate weights
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, h_net.layers.HRangeConv(channels=1, ksize=3, padding='VALID',
                                                            phase=True, in_range=(0,0), out_range=(-2,2), conjugate_weights=False))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(-2,2)))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 5, 2, 1)))
        
        # Test 8: Nested HConv
        in_shape = (6, 6, 1, 1, 1)
        model = get_model(in_shape, 
                          h_net.layers.HRangeConv(channels=1, ksize=3, padding='VALID', name='1',
                                                            phase=True, in_range=(0,0), out_range=(-2,2), conjugate_weights=True),
                          h_net.layers.HRangeConv(channels=1, ksize=3, padding='VALID', name='2',
                                                            phase=True, in_range=(-2,2), out_range=(0,0), conjugate_weights=True))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,0)))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 1, 2, 1)))
    
    def test_HConv(self):
        # Test 1: Default
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, h_net.layers.HConv(channels=1, ksize=3, padding='VALID',
                                                       phase=True, max_order=1, conjugate_weights=True))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,1)))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 2, 2, 1)))
        
        # Test 2: Higher rotation order
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, h_net.layers.HConv(channels=1, ksize=3, padding='VALID',
                                                       phase=True, max_order=2, conjugate_weights=True))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,2)))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 3, 2, 1)))
        
        # Test 3: Nested
        in_shape = (6, 6, 1, 1, 1)
        model = get_model(in_shape, 
                          h_net.layers.HConv(channels=1, ksize=3, padding='VALID', name='1',
                                             phase=True, max_order=2, conjugate_weights=True),
                          h_net.layers.HConv(channels=1, ksize=3, padding='VALID', name='2',
                                             phase=True, max_order=0, conjugate_weights=True))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,0)))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 1, 2, 1)))
               
               
    def test_HBatchNorm(self):
        # Test 1: On one rotation order channel
        in_shape = (4, 4, 1, 2, 1)
        model = get_model(in_shape, h_net.layers.HBatchNorm(keras.activations.relu))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,0)))
        self.assertTrue(check_shape(model, in_shape, out_shape=in_shape))
        
    def test_HBatchNorm_HConv(self):
        # Test 2: On multiple rotation orders
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, 
                          h_net.layers.HConv(channels=1, ksize=3, padding='VALID', max_order=1),
                          h_net.layers.HBatchNorm(keras.activations.relu))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,1)))
        self.assertTrue(check_shape(model, in_shape, (2,2,2,2,1)))
        
    def test_HNonLin(self):
        # Test 1: On one rotation order channel
        in_shape = (4, 4, 1, 2, 1)
        model = get_model(in_shape, h_net.layers.HNonLin(keras.activations.relu))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,0)))
        self.assertTrue(check_shape(model, in_shape, out_shape=in_shape))
        
    def test_HNonLin_HConv(self):
        # Test 2: On multiple rotation orders
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, 
                          h_net.layers.HConv(channels=1, ksize=3, padding='VALID', max_order=1),
                          h_net.layers.HNonLin(keras.activations.relu))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,1)))
        self.assertTrue(check_shape(model, in_shape, (2,2,2,2,1)))
        
    def test_MeanPool(self):
        # Test 1: On one rotation order channel
        in_shape = (4, 4, 1, 2, 1)
        model = get_model(in_shape, h_net.layers.MeanPool(ksize=(2,2), strides=(2,2), padding='VALID'))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,0)))
        self.assertTrue(check_shape(model, in_shape, (2,2,1,2,1)))
        
    def test_MeanPool_HConv(self):
        # Test 2: On multiple rotation orders
        in_shape = (6, 6, 1, 1, 1)
        model = get_model(in_shape, 
                          h_net.layers.HConv(channels=1, ksize=3, padding='VALID', max_order=1),
                          h_net.layers.MeanPool(ksize=(2,2), strides=(2,2), padding='VALID'))
        
        self.assertTrue(check_complex_equivariance(model, in_shape, out_orders=(0,1)))
        self.assertTrue(check_shape(model, in_shape, (2,2,2,2,1)))
         
    def test_Magnitudes(self):
        # Test 1: On one rotation order channel and keepdims=True
        in_shape = (4, 4, 1, 2, 3) # note: multiple channels
        model = get_model(in_shape, h_net.layers.Magnitudes(keepdims=True))
        
        self.assertTrue(check_real_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (4, 4, 1, 1, 3)))
        
        # Test 2: On one rotation order channel and keepdims=False
        in_shape = (4, 4, 1, 2, 3) # note: multiple channels
        model = get_model(in_shape, h_net.layers.Magnitudes(keepdims=False))
        
        self.assertTrue(check_real_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (4, 4, 1, 3)))
        
    def test_Magnitudes_HConv(self):
        # Test 3: On multiple rotation orders and keepdims=True
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, 
                          h_net.layers.HConv(channels=3, ksize=3, padding='VALID', max_order=1), # note: multiple channels
                          h_net.layers.Magnitudes(keepdims=True))
        
        self.assertTrue(check_real_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 2, 1, 3)))
        
        # Test 4: On multiple rotation orders and keepdims=False
        in_shape = (4, 4, 1, 1, 1)
        model = get_model(in_shape, h_net.layers.HConv(channels=3, ksize=3, padding='VALID', max_order=1), # note: multiple channels
                          h_net.layers.Magnitudes(keepdims=False))
        
        self.assertTrue(check_real_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 2, 3)))
        
    def test_Reduce(self):
        # Test 1: On one rotation order channel and keepdims=True
        in_shape = (4, 4, 1, 2, 3) # note: multiple channels
        model = get_model(in_shape, h_net.layers.Reduce(tf.reduce_max, keepdims=True))
        
        self.assertTrue(check_real_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (1, 1, 1, 1, 3)))
        
        # Test 2: keepdims=False
        in_shape = (4, 4, 1, 2, 3) # note: multiple channels
        model = get_model(in_shape, h_net.layers.Reduce(tf.reduce_max, keepdims=False))
        
        self.assertTrue(check_real_equivariance(model, in_shape, out_rot_axes=None))
        self.assertTrue(check_shape(model, in_shape, (3,)))
        
        # Test 3: with complex dimension of input collapsed (Magnitudes)
        in_shape = (4, 4, 1, 3) # note: multiple channels
        model = get_model(in_shape, h_net.layers.Reduce(tf.reduce_max, keepdims=False))
        
        self.assertTrue(check_real_equivariance(model, in_shape, out_rot_axes=None))
        self.assertTrue(check_shape(model, in_shape, (3,)))

if __name__ == '__main__':
    unittest.main()
    
