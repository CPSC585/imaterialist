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

        input_model = model.output

        # size 64
        block1_pad1 = layers.ZeroPadding2D((1, 1), input_model)
        block1_conv1 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(block1_pad1)
        block1_pad2 = layers.ZeroPadding2D((1, 1), block1_conv1)
        block1_conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(block1_pad2)
        block1_maxpool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block1_conv2)

        # size 128
        block2_pad1 = layers.ZeroPadding2D((1, 1), block1_maxpool)
        block2_conv1 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(block2_pad1)
        block2_pad2 = layers.ZeroPadding2D((1, 1), block2_conv1)
        block2_conv2 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(block2_pad2)
        block2_maxpool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block2_conv2)

        # size 256
        block3_pad1 = layers.ZeroPadding2D((1, 1), block2_maxpool)
        block3_conv1 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(block3_pad1)
        block3_pad2 = layers.ZeroPadding2D((1, 1), block3_conv1)
        block3_conv2 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(block3_pad2)
        block3_pad3 = layers.ZeroPadding2D((1, 1), block3_conv2)
        block3_conv3 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(block3_pad3)
        block3_maxpool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block3_conv3)

        # size 512
        block4_pad1 = layers.ZeroPadding2D((1, 1), block3_maxpool)
        block4_conv1 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(block4_pad1)
        block4_pad2 = layers.ZeroPadding2D((1, 1), block4_conv1)
        block4_conv2 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(block4_pad2)
        block4_pad3 = layers.ZeroPadding2D((1, 1), block4_conv2)
        block4_conv3 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(block4_pad3)
        block4_maxpool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block4_conv3)

        # size 512
        block5_pad1 = layers.ZeroPadding2D((1, 1), block4_maxpool)
        block5_conv1 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(block5_pad1)
        block5_pad2 = layers.ZeroPadding2D((1, 1), block5_conv1)
        block5_conv2 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(block5_pad2)
        block5_pad3 = layers.ZeroPadding2D((1, 1), block5_conv2)
        block5_conv3 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu')(block5_pad3)
        block5_maxpool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block5_conv3)

        # Fully Connected layer
        fc_flatten = layers.Flatten()(block5_maxpool)
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
