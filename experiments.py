import tensorflow as tf
from tensorflow import keras

import numpy as np

import h5py
import os

import rb_equivariant_cnn as conv
import rb_equivariant_gcnn as gconv
import rb_equivariant_se2ncnn as dn_conv

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'




RB_CHANNELS = 4
HORIZONTAL_SIZE = 48
HEIGHT = 32

BATCH_SIZE = 8

SIMULATION_NAME = '48_48_32_2000_0.71_0.01_0.3_1000.2'




sim_file = os.path.join('data', f'{SIMULATION_NAME}.h5')

class generator:
    def __init__(self, filename):
        self.filename = filename

    def __call__(self):
        with h5py.File(self.filename, 'r') as hf:
            for snap in hf['data']:
                yield snap, snap

dataset = tf.data.Dataset.from_generator(
     generator(sim_file),
     output_signature=(
         tf.TensorSpec(shape=(HORIZONTAL_SIZE, HORIZONTAL_SIZE, HEIGHT, RB_CHANNELS), dtype=tf.float64),
         tf.TensorSpec(shape=(HORIZONTAL_SIZE, HORIZONTAL_SIZE, HEIGHT, RB_CHANNELS), dtype=tf.float64)))

# dataset = dataset.shuffle(10, reshuffle_each_iteration=True)
dataset = dataset.batch(BATCH_SIZE, False)



G = 'D4' # 'C4' for rotations or 'D4' for rotations and reflections

model = keras.Sequential([
            keras.layers.InputLayer(shape=(HORIZONTAL_SIZE, HORIZONTAL_SIZE, HEIGHT, RB_CHANNELS),
                                    batch_size=BATCH_SIZE),
            
            # add transformation dimension
            keras.layers.Reshape((HORIZONTAL_SIZE, HORIZONTAL_SIZE, 1, HEIGHT, RB_CHANNELS)), 
            
            ###############
            #   Encoder   #
            ###############
            gconv.RB3D_G_Conv('Z2', G, h_ksize=3, v_ksize=5, channels=RB_CHANNELS, h_padding='WRAP', v_padding='SAME', strides=(1,1,1), 
                              name=f'En_Lift_{G}_Conv1'),
            # gconv.BatchNorm(name='En_BN1'),
            keras.layers.Activation('relu', name='En_NonLin1'),
            
            gconv.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID', name='SpatialPool1'),
            keras.layers.Dropout(rate=0.2),
            gconv.RB3D_G_Conv(G, G, h_ksize=3, v_ksize=5, channels=RB_CHANNELS, h_padding='WRAP', v_padding='SAME', strides=(1,1,1), 
                              name=f'En_{G}-Conv2'),
            # gconv.BatchNorm(name='En_BN2'),
            keras.layers.Activation('relu', name='En_NonLin2'),
            
            gconv.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID', name='SpatialPool2'),
            keras.layers.Dropout(rate=0.2),
            gconv.RB3D_G_Conv(G, G, h_ksize=3, v_ksize=5, channels=RB_CHANNELS, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                              name=f'En_{G}-Conv3'),
            # gconv.BatchNorm(name='En_BN3'),
            keras.layers.Activation('relu', name='En_NonLin3'),
            
            gconv.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID', name='SpatialPool3'),
            keras.layers.Dropout(rate=0.2),
            gconv.RB3D_G_Conv(G, G, h_ksize=3, v_ksize=5, channels=RB_CHANNELS, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                              name=f'En_{G}-Conv4'),
            # gconv.BatchNorm(name='En_BN4'),
            keras.layers.Activation('relu', name='En_NonLin4'),
            
            gconv.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID', name='SpatialPool4'),
            
            ###############
            #   Decoder   #
            ###############
            gconv.UpSampling(size=(2,2,2), name='UpSampling1'),
            keras.layers.Dropout(rate=0.2),
            gconv.RB3D_G_Conv(G, G, h_ksize=3, v_ksize=5, channels=RB_CHANNELS, h_padding='WRAP', v_padding='SAME', strides=(1,1,1), 
                              name=f'De_{G}_Conv1'),
            # gconv.BatchNorm(name='De_BN1'),
            keras.layers.Activation('relu', name='De_NonLin1'),
            
            gconv.UpSampling(size=(2,2,2), name='UpSampling2'),
            keras.layers.Dropout(rate=0.2),
            gconv.RB3D_G_Conv(G, G, h_ksize=3, v_ksize=5, channels=RB_CHANNELS, h_padding='WRAP', v_padding='SAME', strides=(1,1,1), 
                              name=f'De_{G}-Conv2'),
            # gconv.BatchNorm(name='De_BN2'),
            keras.layers.Activation('relu', name='De_NonLin2'),
            
            gconv.UpSampling(size=(2,2,2), name='UpSampling3'),
            keras.layers.Dropout(rate=0.2),
            gconv.RB3D_G_Conv(G, G, h_ksize=3, v_ksize=5, channels=RB_CHANNELS, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                              name=f'De_{G}-Conv3'),
            # gconv.BatchNorm(name='De_BN3'),
            keras.layers.Activation('relu', name='De_NonLin3'),
            
            gconv.UpSampling(size=(2,2,2), name='UpSampling4'),
            keras.layers.Dropout(rate=0.2),
            gconv.RB3D_G_Conv(G, G, h_ksize=3, v_ksize=5, channels=RB_CHANNELS, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                              name=f'De_{G}-Conv4'),
            gconv.TransformationPooling(tf.reduce_mean, keepdims=False)
        ])

# output shape: batch_size, width, depth, height, channels
model.summary()



model.compile(
    loss=tf.keras.losses.MeanSquaredError, 
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=["mse"]
)

hist = model.fit(dataset, epochs=100)