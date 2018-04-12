from __future__ import print_function
import os
import glob
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


def read_aug_params(opts, log):
    if opts.aug_params:
        log.debug("Loading Following Augmentation Parameters...")
        with open(opts.aug_params, 'r') as stream:
            data_loaded = yaml.load(stream)
            opts.aug_params = data_loaded
            for key, value in opts.aug_params.iteritems():
                log.info("param {}: {}".format(key, value))
    else:
        log.warn("NO Augmentation, parameters not found!")
    return opts


def read_freeze_layers(opts, log):
    if opts.unfreeze_layers:
        log.debug("Un-Freezing All Layers Except...")
        with open(opts.unfreeze_layers, 'r') as f:
            opts.unfreeze_layers = f.readlines()
            for l in opts.unfreeze_layers:
                log.info(l)
    return opts


def setup_paths(opts, base_dir, log):
    # TODAY.strftime("%d-%b-%Y")
    opts.output_path = os.path.join(base_dir, opts.model, "opt-{}_s{}".format(opts.optimizer, opts.seed))
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


def import_model(opts, log):
    opts.model = importmod(opts.model, "Model {} does not exist!".format(opts.model))


def get_saved_models(opts):
    files = []
    if os.path.exists(opts.output_path):
        files = glob.glob(os.path.join(opts.output_path, 'weights.*.hdf5'))
        files.sort(key=lambda x: float(os.path.basename(x).split("-")[0].replace("weights.", "")))
        return files
