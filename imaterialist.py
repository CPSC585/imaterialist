from __future__ import print_function
import logging
import os
import sys
import datetime
sys.path.insert(0, 'src')
import argparse
from keras.utils import plot_model
from keras.models import load_model
from utils import exists, create_paths, check_opts, import_model, read_val_aug_params
from utils import setup_paths, read_train_aug_params, read_unfreeze_layers, get_saved_models
from iterators import get_std_iterator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TerminateOnNaN, TensorBoard

LEARNING_RATE = 1e-3
NUM_EPOCHS = 60
BATCH_SIZE = 32
NUM_GPUS = 1
OPTIMIZER='nadam' #rmsprop
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
    
    parser.add_argument('--pretrained',
                        dest='pretrained',
                        action='store_true',
                        help='will load pretrained weights from imagenet',
                        default=False)

    parser.add_argument('--aug-val',
                        dest='aug_val',
                        action='store_true',
                        help='augment validation images',
                        default=False)
    
    parser.add_argument('--aug-train',
                        dest='aug_train',
                        action='store_true',
                        help='augment train images',
                        default=False)

    parser.add_argument('--freeze-layers',
                        dest='freeze_layers',
                        action='store_true',
                        help='unfreeze layers specified in unfreeze_layers file in each network implementation, while freezing everything else',
                        default=False)

    parser.add_argument('--seed', type=int,
                        dest='seed',
                        help="random number seed",
                        default="1977")
    return parser


def train(opts):
    pass


if __name__ == '__main__':
    # parse options and setup paths
    parser = build_parser()
    options = parser.parse_args()
    # Create output directory
    options = setup_paths(options, BASE_OUTPUT_DIR, logging)
    logging.basicConfig(filename=os.path.join(options.log_path, 'train.log'), level=logging.DEBUG)
    # Check validity of options
    check_opts(options, logging)
    # Read augmentation parameters
    options = read_train_aug_params(options, logging)
    options = read_val_aug_params(options, logging)
    # Read freeze/unfreeze layer information
    options = read_unfreeze_layers(options, logging)
    # Check if previously run models exist
    saved_models = get_saved_models(options)
    # Get model
    import_model(options)
    network_model = options.model.Network()
    logging.debug("Creating model: {}".format(options.model))
    if options.pretrained:
        logging.info("Loading Pretrained Weights from Imagenet")
    network = network_model.get_network(options)
    if saved_models:
        # Get the best saved model
        logging.debug("Restarting training from saved model {}".format(saved_models[0]))
        network.load_weights(saved_models[0])    
    network.summary()
    logging.info("--- Network ---")
    for l in network.layers:
        logging.info("{} (Trainable = {})".format(l.name, l.trainable))

    # Get data iterators
    logging.info("Creating data iterators for Training and Validation Datasets")
    train_dataiter = get_std_iterator(options.train_aug_params)
    val_dataiter = get_std_iterator(options.val_aug_params)
    # Create Generators
    train_generator = train_dataiter.flow_from_directory(
        options.train_path, seed=options.seed, class_mode='categorical',  color_mode='rgb',
        batch_size=options.batch_size, target_size=(448, 448), shuffle=True)
    
    val_generator = val_dataiter.flow_from_directory(
        options.val_path, seed=options.seed, class_mode='categorical', color_mode='rgb',
        batch_size=options.batch_size, target_size=(448, 448), shuffle=False)

    # Callbacks
    logging.info("Setting up Callbacks")
    callbacks_list = [
        ModelCheckpoint(
            filepath=os.path.join(options.output_path,
                                  "weights_epoch={epoch:02d}-val_loss={val_loss:.2f}.hdf5"),
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
            histogram_freq=0, 
            batch_size=32, 
            write_graph=True, 
            write_grads=True, 
            write_images=True,
        ),
        TerminateOnNaN(),
    ]
    logging.info("Compiling Model...")
    network.compile(optimizer=options.optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'categorical_accuracy'])
    
    logging.info("Training Model...")
    network.fit_generator(train_generator,
                          steps_per_epoch=190000//options.batch_size,
                          epochs=options.epochs,
                          validation_data=val_generator,
                          validation_steps=6245//options.batch_size,
                          callbacks=callbacks_list,
                          verbose=1)
    logging.info("Finished Training!")
