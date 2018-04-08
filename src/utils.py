from __future__ import print_function
import os
import yaml
import importlib


def exists(p, msg):
    assert os.path.exists(p), msg


def importmod(module, msg):
    try:
        return importlib.import_module("{}.{}".format(module, "network"))
    except ImportError:
        print (msg)


def create_paths(opts):
    if not os.path.exists(opts.log_path):
        os.makedirs(opts.log_path)
    
    if not os.path.exists(opts.tf_log_path):
        os.makedirs(opts.tf_log_path)


def read_aug_params(opts):
    if opts.aug_params:
        with open(opts.aug_params, 'r') as stream:
            data_loaded = yaml.load(stream)
            opts.aug_params = data_loaded
    return opts


def read_freeze_layers(opts):
    if opts.unfreeze_layers:
        with open(opts.unfreeze_layers, 'r') as f:
            opts.unfreeze_layers = f.readlines()
    return opts


def setup_paths(opts, base_dir):
    # TODAY.strftime("%d-%b-%Y")
    opts.output_path = os.path.join(base_dir, opts.model,
                                    "lr{}_ep{}_bs{}_opt-{}_s{}".format(opts.learning_rate, opts.epochs,
                                                                      opts.batch_size, opts.optimizer, opts.seed))
    opts.log_path = os.path.join(opts.output_path, 'logs')
    opts.log_file = os.path.join(opts.log_path, 'train.csv')
    opts.tf_log_path = os.path.join(opts.output_path, 'tf_logs')
    create_paths(opts)
    return opts


def check_opts(opts):
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.learning_rate >= 0
    if not os.path.isdir(opts.train_path):
        raise IOError("Train path {} does not exist!".format(opts.train_path))
    if not os.path.isdir(opts.val_path):
        raise IOError("Train path {} does not exist!".format(opts.val_path))
    opts.model = importmod(opts.model, "Model {} does not exist!".format(opts.model))
