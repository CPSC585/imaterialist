# This is a template for sample network
from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.applications.xception import Xception

class Network(object):
    def __init__(self):
        pass
    
    def get_network(self, options):
        model = Xception(include_top=False, input_shape=(448, 448, 3))
        x = model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(128, activation='softmax', name='output')(x)
        return Model(model.input, output)