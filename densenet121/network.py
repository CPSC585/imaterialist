# This is a template for sample network
from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.applications.densenet import DenseNet121

class Network(object):
    def __init__(self):
        pass
    
    def get_network(self, options):
        if options.pretrained:
            model = model = DenseNet121(include_top=False, input_shape=(448, 448, 3), weights='imagenet')
        else:
            model = DenseNet121(include_top=False, input_shape=(448, 448, 3))
        x = model.output
        x = layers.GlobalAveragePooling2D()(x)
        output = layers.Dense(128, activation='softmax', name='output')(x)
        
        if options.freeze_layers:
            for layer in model.layers:
                if any(map(layer.name.startswith, options.unfreeze_layers)):
                    layer.trainable = True
                else:
                    layer.trainable = False

        return Model(model.input, output)
