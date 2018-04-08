# This is a template for sample network
from __future__ import print_function
from keras import layers, Input
from keras.models import Model

class Network(object):
    def __init__(self, kwargs):
        self.unfreeze_layers = kwargs.unfreeze_layers
    
    def get_network(self):
        input_tensor = Input(shape=(448, 448, 1, ))
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name="conv1")(input_tensor)
        x = layers.MaxPooling2D(pool_size=(3, 3), name='maxpool1')(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), name='maxpool2')(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv3')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), name='maxpool3')(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv4')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), name='maxpool4')(x)
        x = layers.Dropout(0.25)(x)
        x = layers.GlobalMaxPooling2D()(x)
        x = layers.Dense(128, activation='relu', name='dense1')(x)
        x = layers.Dropout(0.25)(x)
        output_tensor = layers.Dense(128, activation="softmax", name='output')(x)
        model  = Model(input_tensor, output_tensor)
        return model
