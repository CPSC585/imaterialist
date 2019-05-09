# This is a Dense169 network
from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.applications.densenet import DenseNet169

class Network(object):
    def __init__(self, kwargs):
        self.unfreeze_layers = kwargs.unfreeze_layers
    
    def get_network(self):
        model = DenseNet169(include_top=False, input_shape=(448, 448, 3), pooling='avg')
        x = model.output
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(128, activation='softmax', name='output')(x)
        return Model(model.input, output)
