# This is a template for sample network
from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.applications.vgg16 import VGG16

class Network(object):
    def __init__(self, kwargs):
        self.unfreeze_layers = kwargs.unfreeze_layers
    
    def get_network(self):
        model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
        x = model.output
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output_tensor = layers.Dense(1000, activation="softmax")(x)
        return Model(model.input, output_tensor)
