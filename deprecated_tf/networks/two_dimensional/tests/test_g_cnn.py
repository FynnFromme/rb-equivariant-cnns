import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

import sys, os
tests_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(tests_dir, '..'))

import g_cnn.layers

def get_model(in_shape: tuple, *layers: keras.Layer) -> keras.Model:
    """Creates a tensorflow model with the given input shape and g_cnn.layers.

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


def check_equivariance(model: keras.Model, in_shape: tuple, in_rot_axes: tuple | None = (1,2), 
                       out_rot_axes: tuple | None = (1,2), in_roll_axis: int | None = 3, out_roll_axis: int | None = 3) -> bool:
    """Checks the 90° rotational equivariance of the model.

    Args:
        model (keras.Model): The model to test equivariance of.
        in_shape (tuple): The shape of the input (excluding batch dimension).
        in_rot_axes (tuple | None, optional): The axes of the input to rotate or None if input not rotated. Defaults to (1,2).
        out_rot_axes (tuple | None, optional): The axes of the output to rotated or None if output not rotated. Defaults to (1,2).
        in_roll_axis (int | None, optional): The axis of the input to roll or None if input not rolled. Defaults to 3.
        out_roll_axis (int, | None optional): The axis of the output to roll or None if outptu not rolled. Defaults to 3.

    Returns:
        bool: Whether the model is equivariant under 90° rotations.
    """
    inpt = np.random.randn(1,*in_shape)
    
    inpt_rot = inpt
    if in_rot_axes is not None:
        inpt_rot = np.rot90(inpt_rot, k=1, axes=in_rot_axes)
    if in_roll_axis is not None:
        inpt_rot = np.roll(inpt_rot, shift=1, axis=in_roll_axis)
    
    output = model(inpt, training=False)
    output_rot = model(inpt_rot, training=False)

    output_rot_reversed = output_rot
    if out_rot_axes is not None:
        output_rot_reversed = np.rot90(output_rot_reversed, k=-1, axes=out_rot_axes)
    if out_roll_axis is not None:
        output_rot_reversed = np.roll(output_rot_reversed, shift=-1, axis=out_roll_axis)
    
    return np.isclose(output, output_rot_reversed, rtol=1e-5, atol=1e-3).all()


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
    """Checks 90° rotational equivariance as well as the output shape of every G-CNN layer."""
    
    def test_LiftP4Conv(self):
        # Test 1:without bias
        in_shape = (4, 4, 1, 1)
        model = get_model(in_shape, g_cnn.layers.LiftP4Conv(channels=1, ksize=3, padding='VALID', use_bias=False))
        
        self.assertTrue(check_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 4, 1)))
        
        # Test 2:with bias
        in_shape = (4, 4, 1, 1)
        model = get_model(in_shape, g_cnn.layers.LiftP4Conv(channels=1, ksize=3, padding='VALID', use_bias=True))
        
        self.assertTrue(check_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 4, 1)))
        
    def test_P4Conv(self):
        # Test 1: without bias
        in_shape = (4, 4, 4, 1)
        model = get_model(in_shape, g_cnn.layers.P4Conv(channels=1, ksize=3, padding='VALID', use_bias=False))
        
        self.assertTrue(check_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 4, 1)))
        
        # Test 2: with bias
        in_shape = (4, 4, 4, 1)
        model = get_model(in_shape, g_cnn.layers.P4Conv(channels=1, ksize=3, padding='VALID', use_bias=True))
        
        self.assertTrue(check_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (2, 2, 4, 1)))
        
    def test_NonLin(self):
        in_shape = (4, 4, 4, 1)
        model = get_model(in_shape, keras.layers.Activation(keras.activations.relu))
        
        self.assertTrue(check_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, in_shape))
        
    def test_BatchNormalization(self):
        in_shape = (4,4,4,1)
        model = get_model(in_shape, g_cnn.layers.BatchNormalization())
        
        self.assertTrue(check_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, in_shape))
        
    def test_TransformationPooling(self):
        # Test 1: with spatial dimensions and keepdims=True
        in_shape = (4,4,4,1)
        model = get_model(in_shape, g_cnn.layers.TransformationPooling(tf.reduce_max, keepdims=True))
        
        self.assertTrue(check_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (4,4,1,1)))
        
        # Test 2: with spatial dimensions keepdims=False
        in_shape = (4,4,4,1)
        model = get_model(in_shape, g_cnn.layers.TransformationPooling(tf.reduce_max, keepdims=False))
        
        self.assertTrue(check_equivariance(model, in_shape, out_roll_axis=None))
        self.assertTrue(check_shape(model, in_shape, (4,4,1)))
        
        # Test 3: without spatial dimensions and keepdims=True
        in_shape = (4,1)
        model = get_model(in_shape, g_cnn.layers.TransformationPooling(tf.reduce_max, keepdims=True))
        
        self.assertTrue(check_equivariance(model, in_shape, 
                                           in_rot_axes=None, in_roll_axis=1, out_rot_axes=None, out_roll_axis=1))
        self.assertTrue(check_shape(model, in_shape, (1,1)))
        
        # Test 4: without spatial dimensions keepdims=False
        in_shape = (4,1)
        model = get_model(in_shape, g_cnn.layers.TransformationPooling(tf.reduce_max, keepdims=False))
        
        self.assertTrue(check_equivariance(model, in_shape, 
                                           in_rot_axes=None, in_roll_axis=1, out_rot_axes=None, out_roll_axis=None))
        self.assertTrue(check_shape(model, in_shape, (1,)))
        
    def test_SpatialPooling(self):
        # Test 1: With transformation channel
        in_shape = (4,4,4,1)
        model = get_model(in_shape, g_cnn.layers.SpatialPooling(ksize=(2,2), strides=(2,2), padding='VALID'))
        
        self.assertTrue(check_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (2,2,4,1)))
        
        # Test 2: Without transformation channel
        in_shape = (4,4,1)
        model = get_model(in_shape, g_cnn.layers.SpatialPooling(ksize=(2,2), strides=(2,2), padding='VALID'))
        
        self.assertTrue(check_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (2,2,1)))
        
    def test_ReduceSpatial(self):
        # Test 1: with transformation dimensions and keepdims=True
        in_shape = (4,4,4,1)
        model = get_model(in_shape, g_cnn.layers.ReduceSpatial(tf.reduce_max, keepdims=True))
        
        self.assertTrue(check_equivariance(model, in_shape))
        self.assertTrue(check_shape(model, in_shape, (1,1,4,1)))
        
        # Test 2: with transformation dimensions keepdims=False
        in_shape = (4,4,4,1)
        model = get_model(in_shape, g_cnn.layers.ReduceSpatial(tf.reduce_max, keepdims=False))
        
        self.assertTrue(check_equivariance(model, in_shape, out_rot_axes=None, out_roll_axis=1))
        self.assertTrue(check_shape(model, in_shape, (4,1)))
        
        # Test 3: without transformation dimensions and keepdims=True
        in_shape = (4,4,1)
        model = get_model(in_shape, g_cnn.layers.ReduceSpatial(tf.reduce_max, keepdims=True))
        
        self.assertTrue(check_equivariance(model, in_shape, in_roll_axis=None, out_roll_axis=None))
        self.assertTrue(check_shape(model, in_shape, (1,1,1)))
        
        # Test 4: without transformation dimensions keepdims=False
        in_shape = (4,4,1)
        model = get_model(in_shape, g_cnn.layers.ReduceSpatial(tf.reduce_max, keepdims=False))
        
        self.assertTrue(check_equivariance(model, in_shape, in_roll_axis=None, out_rot_axes=None, out_roll_axis=None))
        self.assertTrue(check_shape(model, in_shape, (1,)))
        

if __name__ == '__main__':
    unittest.main()
    
