from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, 'src')
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import argparse
from keras.models import load_model
from utils import get_models_eval, _read_aug_params
from iterators import get_std_iterator


def build_parser():
    parser = argparse.ArgumentParser(description='iMaterialist Kaggle Competition Evaluate')

    parser.add_argument('--model', type=str,
                        dest='model',
                        help='string path to the Keras model',
                        required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path',
                        help='string path to the train dataset',
                        required=True)

    parser.add_argument('--val-path', type=str,
                        dest='val_path',
                        help='string path to the validation dataset',
                        required=True)
    
    parser.add_argument('--augment',
                        action='store_true',
                        dest='augment',
                        help='Augment evalulation',
                        default=False)

    parser.add_argument('--augment-iter', type=int,
                        dest='augment_iter',
                        help='Number of iterations for augmented evalulation',
                        default=10)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()
    saved_params = get_models_eval(options.model)
    if options.augment:
        augment_path = os.path.join(options.model, 'augment.yml')
        options.aug_params = _read_aug_params(augment_path)
    else:
        options.aug_params = {}
        options.augment_iter = 1
    results = {'filename': [], 'class': [], 'class_index': []}

    _dataiter = get_std_iterator({})
    _generator = _dataiter.flow_from_directory(options.train_path, class_mode='categorical',  color_mode='rgb',
                                               batch_size=1, target_size=(448, 448), shuffle=False)
    class_indices = dict((v, k)
                         for k, v in _generator.class_indices.iteritems())
    for cn in class_indices.values():
        results[cn] = []


for param in saved_params:
    dataiter = get_std_iterator(options.aug_params)
    generator = dataiter.flow_from_directory(options.val_path, class_mode='categorical',  color_mode='rgb',
                                             batch_size=1, target_size=(448, 448), shuffle=False)
    output_file = os.path.splitext(param)[0] + "_" + options.val_path.replace("/", '_') + '_preds.csv'
    model = load_model(param)
    for iteration in range(options.augment_iter):
        probabilities = model.predict_generator(generator, steps=len(generator.filenames), use_multiprocessing=True)
        for i, (f, c) in enumerate(zip(generator.filenames, generator.classes)):
            results['filename'].append(f)
            results['class'].append(class_indices[c])
            results['class_index'].append(c)
            p = probabilities[i]
            for k in class_indices.keys():
                results[class_indices[k]].append(p[k])
    df = pd.DataFrame.from_dict(results)
    df.to_csv(output_file)

