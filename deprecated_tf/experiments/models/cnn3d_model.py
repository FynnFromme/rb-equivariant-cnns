from tensorflow import keras

L2 = 0
# TODO: WRAP PADDING

def build(horizontal_size, height, rb_channels, batch_size):
    model = keras.Sequential([
            keras.layers.InputLayer(shape=(horizontal_size, horizontal_size, height, rb_channels),
                                    batch_size=batch_size),
            
            ##########################
            #    Data Augmentation   #
            ##########################
            keras.layers.Reshape((horizontal_size, horizontal_size, height*rb_channels)),
            keras.layers.RandomRotation(factor=1, fill_mode='wrap', interpolation='bilinear'),
            keras.layers.RandomFlip(mode='horizontal_and_vertical'),
            keras.layers.Reshape((horizontal_size, horizontal_size, height, rb_channels)),
            
            ###############
            #   Encoder   #
            ###############
            keras.layers.Conv3D(kernel_size=(3,3,3), filters=4, padding='same', strides=(1,1,1),
                           use_bias=False, kernel_initializer='he_normal', 
                           kernel_regularizer=keras.regularizers.L2(L2)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            
            keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid'),
            
            keras.layers.Dropout(rate=0.2),
            keras.layers.Conv3D(kernel_size=(3,3,3), filters=8, padding='same', strides=(1,1,1),
                           use_bias=False, kernel_initializer='he_normal', 
                           kernel_regularizer=keras.regularizers.L2(L2)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            
            keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid'),
            
            keras.layers.Dropout(rate=0.2),
            keras.layers.Conv3D(kernel_size=(3,3,3), filters=16, padding='same', strides=(1,1,1),
                           use_bias=False, kernel_initializer='he_normal', 
                           kernel_regularizer=keras.regularizers.L2(L2)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            
            keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid'),
            
            keras.layers.Dropout(rate=0.2),
            keras.layers.Conv3D(kernel_size=(3,3,3), filters=32, padding='same', strides=(1,1,1),
                           use_bias=False, kernel_initializer='he_normal', 
                           kernel_regularizer=keras.regularizers.L2(L2)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            
        #     cnn.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='valid'),
            
            ###############
            #   Decoder   #
            ###############
        #     cnn.UpSampling(size=(2,2,2)),
            
            keras.layers.Dropout(rate=0.2),
            keras.layers.Conv3D(kernel_size=(3,3,3), filters=32, padding='same', strides=(1,1,1),
                           use_bias=False, kernel_initializer='he_normal', 
                           kernel_regularizer=keras.regularizers.L2(L2)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            
            keras.layers.UpSampling3D(size=(2,2,2)),
            
            keras.layers.Dropout(rate=0.2),
            keras.layers.Conv3D(kernel_size=(3,3,3), filters=16, padding='same', strides=(1,1,1),
                           use_bias=False, kernel_initializer='he_normal', 
                           kernel_regularizer=keras.regularizers.L2(L2)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            
            keras.layers.UpSampling3D(size=(2,2,2)),
            
            keras.layers.Dropout(rate=0.2),
            keras.layers.Conv3D(kernel_size=(3,3,3), filters=8, padding='same', strides=(1,1,1),
                           use_bias=False, kernel_initializer='he_normal', 
                           kernel_regularizer=keras.regularizers.L2(L2)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            
            keras.layers.UpSampling3D(size=(2,2,2)),
            
            keras.layers.Dropout(rate=0.2),
            keras.layers.Conv3D(kernel_size=(3,3,3), filters=rb_channels, padding='same', strides=(1,1,1),
                           use_bias=False, kernel_initializer='he_normal', 
                           kernel_regularizer=keras.regularizers.L2(L2)),
        ])
    return model