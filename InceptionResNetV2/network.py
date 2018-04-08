#This is an implementation of InceptionResNetV2
#Allows to freeze specific layers using an array
#from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

class Network(object):
    def __init__(self, **kwargs): 
        pass
    
    def get_network(self, arr):
        #load pre-trained model mobile.net with input (448,448,3)
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        init_model  = InceptionResNetV2(include_top = False, weights='imagenet', input_shape = (448, 448,3), classes = 128)
        x = init_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        output_tensor = Dense(128, activation='softmax')(x)
        
        for layer in init_model.layers:
            if layer.name in arr:
                layer.trainable = False
            
        model = Model(init_model.input, output_tensor)
        return model
