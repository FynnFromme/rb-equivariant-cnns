from tensorflow import keras
from networks.three_dimensional import cnn

L2 = 0

def build(horizontal_size, height, rb_channels, batch_size):
    model = keras.Sequential([
            keras.layers.InputLayer(shape=(horizontal_size, horizontal_size, height, rb_channels),
                                    batch_size=batch_size),
            
            ###############
            #   Encoder   #
            ###############
            cnn.RB3D_Conv(h_ksize=3, v_ksize=3, channels=4, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                           use_bias=False, filter_initializer='he_normal', 
                           filter_regularizer=keras.regularizers.L2(L2)),
            cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            cnn.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID'),
            
            keras.layers.Dropout(rate=0.2),
            cnn.RB3D_Conv(h_ksize=3, v_ksize=3, channels=8, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                           use_bias=False, filter_initializer='he_normal', 
                           filter_regularizer=keras.regularizers.L2(L2)),
            cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            cnn.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID'),
            
            keras.layers.Dropout(rate=0.2),
            cnn.RB3D_Conv(h_ksize=3, v_ksize=3, channels=16, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                           use_bias=False, filter_initializer='he_normal', 
                           filter_regularizer=keras.regularizers.L2(L2)),
            cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            cnn.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID'),
            
            keras.layers.Dropout(rate=0.2),
            cnn.RB3D_Conv(h_ksize=3, v_ksize=3, channels=32, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                           use_bias=False, filter_initializer='he_normal', 
                           filter_regularizer=keras.regularizers.L2(L2)),
            cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
        #     cnn.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID'),
            
            ###############
            #   Decoder   #
            ###############
        #     cnn.UpSampling(size=(2,2,2)),
            
            keras.layers.Dropout(rate=0.2),
            cnn.RB3D_Conv(h_ksize=3, v_ksize=3, channels=32, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                           use_bias=False, filter_initializer='he_normal', 
                           filter_regularizer=keras.regularizers.L2(L2)),
            cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            cnn.UpSampling(size=(2,2,2)),
            
            keras.layers.Dropout(rate=0.2),
            cnn.RB3D_Conv(h_ksize=3, v_ksize=3, channels=16, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                           use_bias=False, filter_initializer='he_normal', 
                           filter_regularizer=keras.regularizers.L2(L2)),
            cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            cnn.UpSampling(size=(2,2,2)),
            
            keras.layers.Dropout(rate=0.2),
            cnn.RB3D_Conv(h_ksize=3, v_ksize=3, channels=8, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                           use_bias=False, filter_initializer='he_normal', 
                           filter_regularizer=keras.regularizers.L2(L2)),
            cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            cnn.UpSampling(size=(2,2,2)),
            
            keras.layers.Dropout(rate=0.2),
            cnn.RB3D_Conv(h_ksize=3, v_ksize=3, channels=rb_channels, h_padding='WRAP', v_padding='SAME', strides=(1,1,1),
                           use_bias=False, filter_initializer='he_normal', 
                           filter_regularizer=keras.regularizers.L2(L2)),
        ])
    return model