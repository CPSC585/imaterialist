from __future__ import print_function
import os
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

