#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:44:31 2018

@author: anushree-ankola
"""

from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.applications.resnet50 import ResNet50

class Network(object):
    def __init__(self):
        pass
    
    def get_network(self, options):
        if options.pretrained:
            model = ResNet50(include_top=False, input_shape=(448, 448, 1), weights='imagenet')
         
        else:
            model = ResNet50(include_top=False, input_shape=(448, 448, 1))
            x = model.output
            x = layers.Dropout(0.5)(x)
            output = layers.Dense(128, activation='softmax', name='output')(x)
            
        if options.freeze_layers:
            for layer in model.layers:
                if any(map(layer.name.startswith, options.unfreeze_layers)):
                    layer.trainable = True
                else:
                    layer.trainable = False
                    
        return Model(model.input, output)

