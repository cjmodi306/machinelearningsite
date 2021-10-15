import numpy as np
import warnings

import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Add, GlobalMaxPooling2D, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Input, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow import random
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity

def identity_block(X, f, filters, intializer = glorot_uniform):
    '''Arguments: input tensor
                kernel_size
                filters: integer, number of filters
                stage: integer
     Returns: output tensor for the block'''
    f1, f2, f3 = filters

    X_shortcut = X

    X = Conv2D(f1, 3, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, f, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, 3, padding='same')(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters):
    '''
    :param X: input tensor
    :param f: size of kernel matrix
    :param filters: list containing number of filters
    :return: convolutionalized tensor X
    '''
    f1, f2, f3 = filters

    X_shortcut = X
    X_shortcut = Conv2D(kernel_size=f, filters=f3, padding='same')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = Conv2D(kernel_size= 2, filters=f1, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(kernel_size=f, filters=f2, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(kernel_size=2, filters=f3, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def resnet(input_shape, classes):
    '''
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3 ->
    CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE

    Arguments:  input_shape: shape of the images of the dataset
                classes: integer, number of classes
    Returns: model-- a Model() instance in Keras
    '''

    X_input = Input(input_shape)

    #Stage 1
    X = Conv2D(64, kernel_size=7, padding='same', kernel_initializer=glorot_uniform)(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(3,strides=2)(X)

    #Stage 2
    X = convolutional_block(X, f=3, filters=[64,64,256])
    X = identity_block(X, 3, [64,64,256])
    X = identity_block(X, 3, [64, 64, 256])

    #Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    #Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    #Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    #AveragePooling
    X = AveragePooling2D((2,2))(X)

    #Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform)(X)

    model = Model(inputs=X_input, outputs = X)

    return model