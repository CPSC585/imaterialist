from __future__ import print_function
import numpy as np
import logging
import sys
sys.path.insert(0, 'src')
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from keras.models import load_model
from utils import get_models_eval
from iterators import get_std_iterator

model = load_model(get_models_eval('xception'))
test_dataiter = get_std_iterator({})
test_generator = test_dataiter.flow_from_directory('data/s448_rgb/test', class_mode=None,  color_mode='rgb', batch_size=1, target_size=(448, 448), shuffle=False)
probabilities = model.predict_generator(test_generator, steps=1, use_multiprocessing=True)
print (len(probabilities))
print (len(test_generator.filenames))


