from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator

def get_std_iterator(aug_dict):
    """
    Check this webpage https://faroit.github.io/keras-docs/1.0.6/preprocessing/image/ aug_dict 
    """

    datagen = ImageDataGenerator(
        samplewise_center=aug_dict.get('samplewise_center', False),
        samplewise_std_normalization=aug_dict.get('samplewise_std_normalization', False),
        zca_whitening=aug_dict.get('zca_whitening', False),
        rotation_range=aug_dict.get('rotation_range', 0),
        width_shift_range=aug_dict.get('width_shift_range', 0),
        height_shift_range=aug_dict.get('height_shift_range', 0),
        shear_range=aug_dict.get('shear_range', 0),
        zoom_range=aug_dict.get('zoom_range', 0),
        channel_shift_range=aug_dict.get('channel_shift_range', 0),
        fill_mode=aug_dict.get('fill_mode', 'nearest'),
        horizontal_flip=aug_dict.get('horizontal_flip', False),
        vertical_flip=aug_dict.get('vertical_flip', False),
        rescale=aug_dict.get('rescale', 0),
        data_format=aug_dict.get('data_format', 'channels_last')
    )  
    return datagen
