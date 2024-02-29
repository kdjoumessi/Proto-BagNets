import os
import sys
import yaml
import socket
import argparse
from tqdm import tqdm
from munch import munchify 
from datetime import datetime 

import torch
from torch.utils.data import DataLoader

def parse_config():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        '-config',
        type=str,
        default='./configs/default.yaml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-print_config',
        action='store_true',
        default=False,
        help='Print details of configs.'
    )
    args = parser.parse_args()
    return args

def load_config(args):
    with open(args.config, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    return munchify(cfg)

def model_save_path(cfg):

    timestamp_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    model_dir = os.path.join(cfg.paths.save, cfg.paths.save, cfg.paths.model_dir, 'OCT', timestamp_str)

    cfg.paths.model_save_path = model_dir

    if cfg.base.test:
        cfg.paths.model_save_path = model_dir = os.path.join(cfg.paths.save, 'Outputs/tmp', timestamp_str)
        
    return cfg

def print_dataset_info(datasets):
    train_dataset, test_dataset, val_dataset = datasets
    print('=========================')
    print('Dataset Loaded.')
    print('Categories:\t{}'.format(len(train_dataset.classes)))
    print('Training:\t{}'.format(len(train_dataset)))
    print('Validation:\t{}'.format(len(val_dataset)))
    print('Test:\t\t{}'.format(len(test_dataset)))
    print('=========================')

def exit_with_error(msg):
    print(msg)
    sys.exit(1)