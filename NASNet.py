#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:25:55 2018

@author: callofduty
"""

from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.applications import NASNetLarge

class Network(object):
    def __init__(self, **kwargs): 
        pass
    
    def get_network(self):
        my_model = NASNetLarge(include_top=False, weights='imagenet')
        input_tensor = Input(shape=(448, 448, 3))
        output_nasnet = my_model(input_tensor)
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(output_nasnet)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output_tensor = layers.Dense(128, activation="softmax")(x)
        model  = Model(input_tensor, output_tensor)        
        return model
