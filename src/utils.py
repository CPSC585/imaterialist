from __future__ import print_function
import os
import sys
import glob
import yaml
from shutil import copyfile
import importlib

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

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

def _read_aug_params(path):
    with open(path, 'r') as stream:
        data_loaded = yaml.load(stream)
    return data_loaded


def read_val_aug_params(opts, log):
    path = os.path.join(opts.model, 'augment.yml')
    if os.path.exists(path) and opts.aug_val:
        copyfile(path, os.path.join(opts.output_path, os.path.basename(path)))
        log.debug("Loading Validation Augmentation Parameters...")
        opts.val_aug_params = _read_aug_params(path)
        for key, value in opts.val_aug_params.iteritems():
            log.info("Validation augmentation param -- {}: {}".format(key, value))
    else:
        opts.val_aug_params = {}
        log.warn("NO Validation Augmentation.")
    return opts


def read_train_aug_params(opts, log):
    path = os.path.join(opts.model, 'augment.yml')
    if os.path.exists(path) and opts.aug_train:
        copyfile(path, os.path.join(opts.output_path, os.path.basename(path)))
        log.debug("Loading Train Augmentation Parameters...")
        opts.train_aug_params = _read_aug_params(path)
        for key, value in opts.train_aug_params.iteritems():
            log.info("Train augmentation param -- {}: {}".format(key, value))
    else:
        opts.train_aug_params = {}
        log.warn("NO Train Augmentation.")
    return opts


def read_unfreeze_layers(opts, log):
    path = os.path.join(opts.model, 'unfreeze_layers')
    if opts.freeze_layers and os.path.exists(path):
        copyfile(path, os.path.join(opts.output_path, os.path.basename(path)))
        with open(path, 'r') as f:
            opts.unfreeze_layers = f.readlines()
            opts.unfreeze_layers = map(lambda s: s.strip(), opts.unfreeze_layers)
    elif opts.freeze_layers and not os.path.exists(path):
        log.warn("Specified freezing layers but no file with information about freeze layers found! All Layers are Trainable.")
    return opts


def setup_paths(opts, base_dir, log):
    # TODAY.strftime("%d-%b-%Y")
    name = "opt-{}_s-{}".format(opts.optimizer, opts.seed)
    if opts.aug_train:
        name += "_aug_train"
    if opts.aug_val:
        name += "_aug_val"
    if opts.pretrained:
        name += "_pretrained"
    if opts.freeze_layers and os.path.exists(os.path.join(opts.model, 'unfreeze_layers')):
        name += "_partly_frozen"
    opts.output_path = os.path.join(base_dir, opts.model, name)
    opts.log_path = os.path.join(opts.output_path, 'logs')
    opts.log_file = os.path.join(opts.log_path, 'train.csv')
    opts.tf_log_path = os.path.join(opts.output_path, 'tf_logs')
    create_paths(opts)
    return opts


def check_opts(opts, log):
    assert opts.epochs > 0
    log.info("Number of Epochs: {}".format(opts.epochs))
    assert opts.batch_size > 0
    log.info("Batch Size: {}".format(opts.batch_size))
    if not os.path.isdir(opts.train_path):
        log.error("Train path {} does not exist!".format(opts.train_path))
        raise IOError("Train path {} does not exist!".format(opts.train_path))
    else:
        log.info("Training Data: {}".format(opts.train_path))
    if not os.path.isdir(opts.val_path):
        log.error("Train path {} does not exist!".format(opts.val_path))
        raise IOError("Train path {} does not exist!".format(opts.val_path))
    else:
        log.info("Validation Data: {}".format(opts.val_path))


def import_model(opts):
    opts.model = importmod(opts.model, "Model {} does not exist!".format(opts.model))


def get_saved_models(opts):
    files = []
    if os.path.exists(opts.output_path):
        files = glob.glob(os.path.join(opts.output_path, '*.hdf5'))
        files.sort(key=lambda x: float(os.path.basename(x).split("-")[1].replace(".hdf5", "").replace("val_loss=", "")))
        return files


def get_models_eval(model):
    files = glob.glob(os.path.join('trained_models', model, '*/*.hdf5'))
    files.sort(key = lambda x: float(os.path.basename(x).split("-")[1].replace(".hdf5", "").replace("val_loss=", "")))
    org_files = {}
    for path in files:
        exp_name = os.path.basename(os.path.dirname(path))
        if not exp_name in org_files.keys():
            org_files[exp_name] = [path]
        else:
            org_files[exp_name].append(path)
    return [x[0] for x in org_files.values()]
    
