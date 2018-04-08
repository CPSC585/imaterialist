from __future__ import print_function
import os
import sys
import datetime
sys.path.insert(0, 'src')
import argparse
from keras.utils import plot_model
from utils import exists, importmod, create_paths, check_opts
from utils import setup_paths, read_aug_params, read_freeze_layers
from iterators import get_std_iterator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TerminateOnNaN, TensorBoard


LEARNING_RATE = 1e-3
NUM_EPOCHS = 60
BATCH_SIZE = 32
NUM_GPUS = 1
OPTIMIZER='nadam'
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

    parser.add_argument('--optimizer', type=str,
                        dest='optimizer',
                        help='all Keras optimizers',
                        metavar='OPTIMIZER', default=OPTIMIZER)
    
    parser.add_argument('--aug-params', type=str,
                        dest='aug_params',
                        help='a yaml dictionary the defines the augmentation kwargs',
                        default=None)

    parser.add_argument('--unfreeze-layers', type=str,
                        dest='unfreeze_layers',
                        help='a text file with names (one per line) of all layers that should NOT be frozen. \
                              Unspecified layers will be frozen. If None all layers will be unfrozen',
                        default=None)

    parser.add_argument('--seed', type=int,
                        dest='seed',
                        help="random number seed",
                        default="2018")
    return parser


def train(opts):
    pass


if __name__ == '__main__':
    # parse options and setup paths
    parser = build_parser()
    options = parser.parse_args()
    # Create output directory
    options = setup_paths(options, BASE_OUTPUT_DIR)
    # Read augmentation parameters
    options = read_aug_params(options)
    # Read freeze/unfreeze layer information
    options = read_freeze_layers(options)
    # Check validity of options
    check_opts(options)
    # Get model, net, iterators
    network_model = options.model.Network(options)
    network = network_model.get_network()
    network.summary()

    # Get data iterators
    if options.aug_params:
        train_dataiter = get_std_iterator(**options.aug_params)
    else:
        train_dataiter = get_std_iterator()
    val_dataiter = get_std_iterator()
    
    # Create Generators
    train_generator = train_dataiter.flow_from_directory(
        options.train_path, seed=options.seed, class_mode='categorical',  color_mode='rgb',
        batch_size=options.batch_size, target_size=(448, 448), shuffle=True)
    val_generator = train_dataiter.flow_from_directory(
        options.val_path, seed=options.seed, class_mode='categorical', color_mode='rgb',
        batch_size=options.batch_size, target_size=(448, 448), shuffle=False)

    # Callbacks
    callbacks_list = [
        ModelCheckpoint(
            filepath=os.path.join(options.output_path,
                                  "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
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
                          steps_per_epoch=190000//options.batch_size,
                          epochs=options.epochs,
                          validation_data=val_generator,
                          validation_steps=6245//options.batch_size,
                          callbacks=callbacks_list,
                          verbose=1)
