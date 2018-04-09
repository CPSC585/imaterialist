# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:54:05 2018

@author: Rishabh Sharma
"""

from __future__ import print_function
from keras import layers,Input
from keras.models import Model
from keras.applications.nasnet import NASNetLarge


class Network(object):
    def __init__(self,kwargs):
        self.unfreeze_layers = kwargs.unfreeze_layers
        
    def get_network(self):
        model = NASNetLarge(include_top = False, input_shape=(448,448,3))
        x = model.output
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation="relu")(x)
        predictions = layers.Dense(128, activation="softmax")(x)
        return Model(model.input, predictions)