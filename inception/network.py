# This is an implementation of InceptionV3
from __future__ import print_function
import keras
from keras import layers, Input
from keras.models import Model
from keras.applications import inception_v3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

class Network(object):
    def __init__(self, **kwargs): 
        pass
    
    def get_network(self, options):
    	if options.pretrained:
        	modelV3 = inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                           input_shape=(448, 448, 3))
        else:
        	modelV3 = inception_v3.InceptionV3(include_top=False,
                                           input_shape=(448, 448, 3))
        x = modelV3.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        output_tensor = Dense(128, activation='softmax' name="predictions")(x)
        

        if options.freeze_layers:
            for layer in modelV3.layers:
            	if any(map(layer.name.startwith, options.unfreeze_layers)):
            		layer.trainable = True
            	else:
            		layer.trainable = False


        model = Model(modelV3.input, output_tensor)
        return model

