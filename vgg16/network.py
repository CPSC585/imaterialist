from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.applications.vgg16 import VGG16


class Network(object):
    def __init__(self, **kwargs): 
        self.options = kwargs.get('options')
    
    def get_network(self):

        
        #Check for pretrained data
        if self.options.pretrianed:
            model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        else:
            model = VGG16(include_top=False, input_shape=(224, 224, 3))
        
        x = model.output
        
        #size 64
        x = layers.ZeroPadding2D((1,1),x)
        x = layers.Conv2D(64, kernel_size=(3,3),activation='relu')(x)
        x = layers.ZeroPadding2D((1,1),x)
        x = layers.Conv2D(64, kernel_size=(3,3),activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        
        #size 128
        x = layers.ZeroPadding2D((1,1),input_tensor)
        x = layers.Conv2D(128, kernel_size=(3,3),activation='relu')(x)
        x = layers.ZeroPadding2D((1,1),x)
        x = layers.Conv2D(128, kernel_size=(3,3),activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

        #size 256
        x = layers.ZeroPadding2D((1,1),input_tensor)
        x = layers.Conv2D(256, kernel_size=(3,3),activation='relu')(x)
        x = layers.ZeroPadding2D((1,1),input_tensor)
        x = layers.Conv2D(256, kernel_size=(3,3),activation='relu')(x)
        x = layers.ZeroPadding2D((1,1),x)
        x = layers.Conv2D(256, kernel_size=(3,3),activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

        #size 512
        x = layers.ZeroPadding2D((1,1),input_tensor)
        x = layers.Conv2D(512, kernel_size=(3,3),activation='relu')(x)
        x = layers.ZeroPadding2D((1,1),input_tensor)
        x = layers.Conv2D(512, kernel_size=(3,3),activation='relu')(x)
        x = layers.ZeroPadding2D((1,1),x)
        x = layers.Conv2D(512, kernel_size=(3,3),activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

        #size 512
        x = layers.ZeroPadding2D((1,1),input_tensor)
        x = layers.Conv2D(512, kernel_size=(3,3),activation='relu')(x)
        x = layers.ZeroPadding2D((1,1),input_tensor)
        x = layers.Conv2D(512, kernel_size=(3,3),activation='relu')(x)
        x = layers.ZeroPadding2D((1,1),x)
        x = layers.Conv2D(512, kernel_size=(3,3),activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        
        #Fully Connected layer
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output_tensor = layers.Dense(4096, activation='softmax')(x)
        
        #Freeze layers
        # if self.options.freeze_layers:
        #     for layer in model.layers:
        #         if any(map(layer.name.startswith, self.options.unfreeze_layers)):
        #             layer.trainable = True
        #         else:
        #             layer.trainable = False
        
        
        return Model(model.input, output_tensor)


