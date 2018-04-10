# This is a template for sample network
from __future__ import print_function
from keras import layers, Input
from keras.models import Model

class Network(object):
    def __init__(self, **kwargs): 
        pass
    
    def get_network(self):
        input_tensor = Input(shape=(224, 224, 3, ))
        x = layers.ZeroPadding2D((1,1),input_tensor)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(x)
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(x)
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(x)
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(x)
        x = layers.ZeroPadding2D((1,1))(x)
        x = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output_tensor = layers.Dense(1000, activation="softmax")(x)
        
#        if weights_path:
#            model.load_weights(weights_path)
#            return model
            
        model  = Model(input_tensor, output_tensor)
        return model
