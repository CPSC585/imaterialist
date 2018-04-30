#This is an implementation of InceptionResNetV2

from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_resnet_v2 import InceptionResNetV2

class Network(object):
    def __init__(self): 
        pass
    
    def get_network(self, options):
        if options.pretrained:
            init_model  = InceptionResNetV2(include_top=False, weights='imagenet', 
                                            input_shape=(448, 448,3))
        else:
            init_model = InceptionResNetV2(include_top=False,
                                           input_shape=(448, 448, 3))
        x = init_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output_tensor = Dense(128, activation='softmax')(x)
		
        if options.freeze_layers:
            for layer in init_model.layers:
                if any(map(layer.name.startswith, options.unfreeze_layers)):
                    layer.trainable = True
                else:
                    layer.trainable = False

        model = Model(init_model.input, output_tensor)
        return model
