import time
import random
import argparse
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from data_loader import *
from train import *
from model import *
from utils import *
from model import *

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='UIL')
    parse.add_argument("--gpu", type=int, default=1, choices=[0, 1, 2, 3])
    parse.add_argument('--times', type=int, default=1, help='times of repeat experiment')
    parse.add_argument('--save', type=str, default="./result/", help='dir of save')
    parse.add_argument('--dense_dataset', type=str, default="tw", choices=['tw','fs','isp'], help='dataset for experiment')
    parse.add_argument('--sparse_dataset', type=str, default="fb", choices=['fs','fb','wb'], help='dataset for experiment')
    parse.add_argument('--epochs', type=int, default=50, help='Number of total epochs')
    parse.add_argument('--batch_size', type=int, default=32, help='Size of one batch')
    parse.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parse.add_argument('--embed_size', type=int, default=64, help='Number of embeding dim')
    parse.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parse.add_argument('--num_layers_mask', type=int, default=4, help='Number of EncoderLayer')
    parse.add_argument('--num_layers_attn', type=int, default=1, help='Number of attention layes')
    parse.add_argument('--num_layers', type=int, default=4, help='Number of EncoderLayer')
    parse.add_argument('--p_drop', type=float, default=0.1, help='Probability of DROPOUT')
    parse.add_argument('--threshold', type=float, default=0.1, help='Threshold of Mask')
    parse.add_argument('--patience', type=int, default=5, help='How long to wait after last time validation loss improved')
    args = parse.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_device(gpu_id=None):
    if gpu_id is not None:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        else:
            raise ValueError("CUDA is not available on this machine.")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

USE_SEED = True

def main():
    args = parse_args()
    os.makedirs(args.save + 'temp/', exist_ok=True)
    logger = Logger(log_dir=args.save + 'log/', log_file='training.log')
    fixed_seeds = [454, 512, 631, 812, 932]

    with open('./dataset/' + args.dense_dataset + '_' + args.sparse_dataset + '.pkl', 'rb') as file:
        data = pickle.load(file)

    train_samples = data['train_samples']
    val_samples = data['val_samples']
    test_samples = data['test_samples']
    loc_size = data['loc_size']
        
    #----------------Repeat the experiment-------------#
    for idx in range(args.times):
        
        #---------------Repeatability settings---------------#
        if USE_SEED == True:
            seed = fixed_seeds[idx]
            set_seed(seed)
            devices = set_device(args.gpu)
            logger.info('The {} round, start training with random seed {}'.format(idx, seed))
        else:
            seed = None
            logger.info('The {} round, start training without random seed'.format(idx))
        
        current_time = time.time()
        #----------Building networks and optimizers----------#
        model = MTLink(args.embed_size, args.embed_size*2, args.num_layers, args.num_layers_mask, args.num_layers_attn, args.p_drop, args.num_heads, args.threshold, devices, loc_size).to(devices)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        #-------------Start training and logging-------------#
        train_model(train_samples, val_samples, model, optimizer, devices, args, logger, seed)
        model.load_state_dict(torch.load(args.save + 'temp/' + args.dense_dataset +'_' + args.sparse_dataset + '_' + str(seed) + '_checkpoint.pt'))
        metric = test_model(test_samples, model, devices, args, logger)
        print(metric)
        logger.info("Total time elapsed: {:.4f}s".format(time.time() - current_time))
        logger.info('Fininsh trainning in seed {}\n'.format(seed))
    print("Finished")
        
if __name__ == '__main__':
    main()