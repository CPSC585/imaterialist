#This is an implementation of InceptionResNetV2

from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_resnet_v2 import InceptionResNetV2

class Network(object):
    def __init__(self, **kwargs): 
        pass
    
    def get_network(self, options):
        #load pre-trained model mobile.net with input (448,448,3)
        init_model  = InceptionResNetV2(include_top=False, weights='imagenet', 
                                        input_shape=(448, 448,3))
        x = init_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        output_tensor = Dense(128, activation='softmax')(x)
		
        #freezing the layers which are not specified in the arr
        if len(options.unfreeze_layers) > 0:
            for layer in init_model.layers:
                if not layer.name in options.unfreeze_layers:
                    layer.trainable = False
                    
        model = Model(init_model.input, output_tensor)
        return model
