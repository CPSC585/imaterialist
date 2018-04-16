from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.applications.vgg16 import VGG16

class Network(object):
    def __init__(self, kwargs):
        self.unfreeze_layers = kwargs.unfreeze_layers
    
    def get_network(self, options):
        if options.pretrained:
            model = VGG16(include_top=False, weights='imagenet', input_shape=(448, 448, 3))
        else:
            model = VGG16(include_top=False, input_shape=(448, 448, 3))
        
        x = model.output
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output_tensor = layers.Dense(128, activation="softmax")(x)
        
        if options.freeze_layers:
                    for layer in model.layers:
                        if any(map(layer.name.startswith, options.unfreeze_layers)):
                            layer.trainable = True
                        else:
                            layer.trainable = False
                            
        return Model(model.input, output_tensor)
