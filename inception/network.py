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
    
    def get_network(self, unfreezeLayers):
        modelV3 = inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                           input_shape=(448, 448, 3), classes=128)
        x = modelV3.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        output_tensor = Dense(128, activation='softmax' name="predictions")(x)
        model  = Model(modelV3.input, output_tensor)
        
        if len(unfreezeLayers) if > 0:
            for layer in modelV3.layers:
                if not layer.name in unfreezeLayers:
                    layer.trainable = False
        
        return model
