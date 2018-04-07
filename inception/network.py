# This is a template for sample network
from __future__ import print_function
import keras
from keras import layers, Input
from keras.models import Model
from keras.applications import inception_v3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

class Network(object):
    def __init__(self, **kwargs): 
        pass
    
    def get_network(self):
        modelV3 = inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                           input_shape=(448, 448, 3), classes=128)
        x = modelV3.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        output_tensor = layers.Dense(128, activation="softmax")(x)
        model  = Model(modelV3.input, output_tensor)
        for layer in modelV3.layers:
            layer.trainable = False
        return model
