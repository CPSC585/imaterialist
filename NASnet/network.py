# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:54:05 2018

@author: Rishabh Sharma,Himanshu Raj, Dipaksingh Bhukwal, Eizat Mushtaq
"""

from __future__ import print_function
from keras import layers,Input
from keras.models import Model
from keras.applications.nasnet import NASNetLarge


class Network(object):
    def __init__(self,kwargs):
        self.unfreeze_layers = kwargs.unfreeze_layers
        
    def get_network(self):
        