from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator

def get_std_iterator(**kwargs):
    """
    Check this webpage https://faroit.github.io/keras-docs/1.0.6/preprocessing/image/ kwargs 
    """
    datagen = ImageDataGenerator(kwargs)
    return datagen
