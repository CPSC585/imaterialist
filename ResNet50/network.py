# This is a template for sample network
from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.applications import ResNet50

class Network(object):
    def __init__(self, kwargs):
        self.unfreeze_layers = kwargs.unfreeze_layers
    
    def get_network(self):
        model = ResNet50(include_top=False, input_shape=(448, 448, 3))
        x = model.output
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(128, activation='softmax', name='output')(x)
        return Model(model.input, output)
