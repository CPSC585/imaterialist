from __future__ import print_function
import os
import sys
import datetime
import yaml
sys.path.insert(0, 'src')
import argparse
from keras.utils import plot_model
from utils import exists, importmod, create_paths
from iterators import get_std_iterator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TerminateOnNaN, TensorBoard


LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 4
NUM_GPUS = 1
LR_MULT = 1e-3
OPTIMIZER='adam'
BASE_OUTPUT_DIR = "trained_models"
TODAY = datetime.date.today()

def build_parser():
    parser = argparse.ArgumentParser(description='iMaterialist Kaggle Competition Trainer')

    parser.add_argument('--train-path', type=str,
                        dest='train_path',
                        help='string path to the train dataset',
                        required=True)

    parser.add_argument('--val-path', type=str,
                        dest='val_path',
                        help='string path to the validation dataset',
                        required=True)

    parser.add_argument('--model', type=str,
                        dest='model',
                        help='string path to the Keras model',
                        required=True)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--num-gpus', type=int,
                        dest='num_gpus', help='batch size',
                        metavar='num_gpus', default=NUM_GPUS)

    parser.add_argument('--optimizer', type=str,
                        dest='optimizer',
                        help='all Keras optimizers',
                        metavar='OPTIMIZER', default=OPTIMIZER)
    
    parser.add_argument('--aug-params', type=str,
                        dest='aug_params',
                        help='a yaml dictionary the defines the augmentation kwargs',
                        default=None)

    parser.add_argument('--seed', type=int,
                        dest='seed',
                        help="random number seed",
                        default="2018")
    return parser


def check_convert_opts(opts):
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.num_gpus > 0
    assert opts.learning_rate >= 0
    assert opts.lr_mult > 0
    opts.model = importmod(opts.model, "Model {} does not exist!".format(opts.model))

def setup_paths(opts):
    # TODAY.strftime("%d-%b-%Y")
    opts.output_path = os.path.join(BASE_OUTPUT_DIR, opts.model,
                                    "lr{}_ep{}_bs{}_opt{}_s{}".format(opts.learning_rate, opts.epochs,
                                                                      opts.batch_size, opts.optimizer, opts.seed))
    opts.log_path = os.path.join(opts.output_path, 'logs')
    opts.log_file = os.path.join(opts.log_path, 'train.csv')
    opts.tf_log_path = os.path.join(opts.output_path, 'tf_logs')
    create_paths(opts)
    return opts

def read_aug_params(opts):
    if opts.aug_params:
        with open(opts.aug_params, 'r') as stream:
            data_loaded = yaml.load(stream)
            opts.aug_params = data_loaded
    return opts

def train(opts):
    pass


if __name__ == '__main__':
    # parse options and setup paths
    parser = build_parser()
    options = parser.parse_args()
    options = setup_paths(options)
    options = read_aug_params(options)
    check_convert_opts(options)
    # Get model, net, iterators
    network_model = options.model.Network()
    network = network_model.get_network()
    network.summary()

    # Get data iterators
    train_dataiter = get_std_iterator(**options.aug_params)
    val_dataiter = get_std_iterator()
    
    # Create Generators
    train_generator = train_dataiter.flow_from_directory(
        options.train_path, seed=options.seed, class_mode='categorical',  color_mode='grayscale',
        batch_size=options.batch_size, target_size=(448, 448), shuffle=True)
    val_generator = train_dataiter.flow_from_directory(
        options.val_path, seed=options.seed, class_mode='categorical', color_mode='grayscale',
        batch_size=options.batch_size, target_size=(448, 448), shuffle=False)

    # Callbacks

    callbacks_list = [
        ModelCheckpoint(
            filepath = os.path.join(options.output_path, "model.h5"),
            save_best_only=True,
            monitor='val_loss'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=2
        ),
        CSVLogger(
            options.log_file,
            separator=',', 
            append=True,
        ),
        TensorBoard(
            log_dir=options.tf_log_path,
            histogram_freq=1, 
            batch_size=32, 
            write_graph=True, 
            write_grads=True, 
            write_images=True,
        ),
        TerminateOnNaN(),
    ]

    network.compile(optimizer=options.optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'categorical_accuracy'])
    
    network.fit_generator(train_generator,
                          epochs=options.epochs,
                          validation_data=val_generator,
                          callbacks=callbacks_list,
                          verbose=1)