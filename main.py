import os
import re
import time
import shutil
import random
import argparse
import numpy as np
import torch, gc
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from train import train, evaluate
from utils.helpers import makedir
from utils.metrics import Estimator
from utils.log import create_logger
from data.builder import generate_dataset
from modules.builder import generate_model
from utils.func import parse_config, model_save_path, load_config

torch.autograd.set_detect_anomaly(True)

def main():
    torch.cuda.empty_cache()
    gc.collect()

    # load conf with paths
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0') 
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    #print('cuda device number: ', os.environ['CUDA_VISIBLE_DEVICES'])

    # read the config (default.yaml) 
    args = parse_config()
    cfg = load_config(args)  
    cfg = model_save_path(cfg) 

    set_random_seed(cfg.base.random_seed)
    cfg.base.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    base_architecture_type = re.match('^[a-z]*', cfg.train.base_architecture).group(0)
    
    print('device: ', cfg.base.device)
    print('Proto shape: ', cfg.prototype.shape)
    print('base architecture: ', base_architecture_type)
    
    # save training files and settings
    folders = ['data', 'modules', 'training', 'utils']
    
    model_dir = cfg.paths.model_save_path
    cfg.paths.img_dir = os.path.join(model_dir, 'img')   
    save_file = os.path.join(model_dir, 'save_files')

    # creating output dir
    makedir(save_file)
    makedir(cfg.paths.model_save_path) 
    makedir(cfg.paths.img_dir)
    makedir(os.path.join(model_dir, 'val'))

    ## copy useful files
    copy_file(folders, save_file)
    print(model_dir)   

    logger = SummaryWriter(model_dir)
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

    datasets = generate_dataset(cfg)
    model, multi_model = generate_model(cfg)
    estimators = (Estimator(cfg), Estimator(cfg))

    since = time.time()

    print('batch size: ', cfg.train.batch_size)
    print('Proto shape: ', cfg.prototype.shape)
    print('Regularization: ', cfg.train.reg)
    
    print('Regularization epoch: ', cfg.train.reg_epoch)  
    print(f"Regularization FCL: {cfg.train.reg_fcl} \t reg: {cfg.solver.coefs['l1']}")
    print('Regularization (dissimilarity) Proto: ', cfg.train.reg_proto)  
    
    train(
        cfg=cfg,
        model=model,
        multi_model=multi_model,
        dataset = datasets,
        paths = model_dir,
        estimator=estimators,
        logs=(log, logclose, logger),
        debug= debug or test
    )

    name = 'fcl_best_val_weights'
    save_path = cfg.paths.model_save_path #'./save/inference', 
    _, _, dset_test, val_dataset = datasets
    if cfg.data.binary: # only use for binary task to directly save the performance of the best model on the test set
        model_list = ['acc', 'auc']
        model_name = ['Accuracy', 'AUC']
    else:
        model_list = ['acc', 'kappa'] #['acc'] 'loss'
        model_name = ['Accuracy', 'Kappa'] # ['Accuracy'] 'loss'
        
    for i in range(len(model_list)):
        estimator = Estimator(cfg)
        print('========================================')
        print(f'This is the performance of the final model base on the best {model_name[i]}')
        checkpoint = os.path.join(save_path, f'{name}_{model_list[i]}.pt')
        evaluate(cfg, checkpoint, val_dataset, estimator, type_ds='validation')
        
        print('')
        evaluate(cfg, checkpoint, dset_test, estimator, type_ds='test')
        print('')

    time_elapsed = time.time() - since
    print('Training and evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# copy files: can be more customize
def copy_file(folds, dest_dir):
    for directory in folds:
        src_dir = os.path.join(os.getcwd(), directory)
        dst_dir = os.path.join(dest_dir, directory)
        makedir(dst_dir)
        files = os.listdir(src_dir)
        for file in files:
            fpath = os.path.join(src_dir, file)
            if file.startswith("__"):
                continue
            elif os.path.isdir(fpath):
                fs = os.listdir(fpath)
                for f in fs:
                    if not f.startswith(tuple(["__", "."])):
                        shutil.copy2(os.path.join(fpath, f), dst_dir)
            else:
                shutil.copy2(fpath, dst_dir)
    
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=dest_dir)    
    shutil.copy(src=os.path.join(os.getcwd(), 'train.py'), dst=dest_dir)    
    shutil.copy(src=os.path.join(os.getcwd(), 'configs', 'default.yaml'), dst=dest_dir)  
    
if __name__ == '__main__':
    main()