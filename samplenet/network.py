# This is a template for sample network
from __future__ import print_function
from keras import layers, Input
from keras.models import Model

class Network(object):
    def __init__(self, **kwargs): 
        pass
    
    def get_network(self):
        input_tensor = Input(shape=(448, 448, 1, ))
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(input_tensor)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output_tensor = layers.Dense(128, activation="softmax")(x)
        model  = Model(input_tensor, output_tensor)
        return model
