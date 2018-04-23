# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:54:05 2018

@author: Rishabh Sharma
"""

from __future__ import print_function
from keras import layers,Input
from keras.models import Model
from keras.applications.nasnet import NASNetLarge, NASNetMobile


class Network(object):
    def __init__(self,kwargs):
        pass
        
    def get_network(self,options):
        if options.pretrained:
            init_model = NASNetLarge(include_top = False, weights = 'imagenet', input_shape=(448,448,3))
        else:
            init_model = NASNetLarge(include_top = False, input_shape=(448,448,3))
        
        x = init_model.output
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation="relu")(x)
        predictions = layers.Dense(128, activation="softmax")(x)
        
        if options.freeze_layers:
            for layer in init_model.layers:
                if any(map(layer.name.startswith, options.unfreeze_layers)):
                    layer.trainable = True
                else:
                    layer.trainable = False
        return Model(init_model.input, predictions)