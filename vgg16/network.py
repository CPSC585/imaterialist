from __future__ import print_function
from keras import layers, Input
from keras.models import Model
from keras.applications.vgg16 import VGG16


class Network(object):
    def __init__(self, **kwargs):
        self.options = kwargs.get('options')

    def get_network(self):

        # Check for pre-trained data
        if self.options.pretrianed:
            model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        else:
            model = VGG16(include_top=False, input_shape=(224, 224, 3))

        conv_base_model = model.output
        # Fully Connected layer
        fc_flatten = layers.Flatten()(conv_base_model)
        fc_dense1 = layers.Dense(4096, activation='relu')(fc_flatten)
        fc_dropout1 = layers.Dropout(0.5)(fc_dense1)
        fc_dense2 = layers.Dense(4096, activation='relu')(fc_dropout1)
        fc_dropout2 = layers.Dropout(0.5)(fc_dense2)
        output_tensor = layers.Dense(4096, activation='softmax')(fc_dropout2)

        # Freeze layers
        if self.options.freeze_layers:
            for layer in model.layers:
                if any(map(layer.name.startswith, self.options.unfreeze_layers)):
                    layer.trainable = True
                else:
                    layer.trainable = False

        return Model(model.input, output_tensor)
