import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Add, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from resnets_utils import *
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow

from test_utils import summary, comparator
import public_tests
from main_utils import *
'''
X1 =  np.ones((1,4,4,3))*-1
X2 = np.ones((1,4,4,3))*1
X3 = np.ones((1,4,4,3))*3

X = np.concatenate((X1, X2, X3), axis=0).astype(np.float32)

A= identity_block(X, 3, [10,20,3])
A = convolutional_block(X, 2, [4,4,3])
print(X.shape)'''

model = resnet(input_shape=(64,64,3), classes=6)
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255
X_test = X_test_orig/255

Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

model.fit(X_train, Y_train, epochs=10, batch_size = 32)

preds = model.evaluate(X_test, Y_test)
print("Test accuracy: {}".format(preds[1]))