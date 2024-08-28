import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import logging
import os

class Logger:
    def __init__(self, log_dir='logs', log_file='train.log', log_level=logging.INFO):
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(log_level)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setLevel(log_level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def get_logger(self):
        return self.logger



class EarlyStopping:
    def __init__(self, logger, sparse_dataset_name, dense_dataset_name, seed, patience=5, verbose=False, delta=0, save = None):
        self.logger = logger
        self.patience = patience 
        self.verbose = verbose 
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta 
        self.save = save
        self.seed = seed
        self.sparse_dataset_name = sparse_dataset_name
        self.dense_dataset_name = dense_dataset_name

    def __call__(self, val_loss, model):
        score = -val_loss 
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(model.state_dict(), self.save + 'temp/'+ self.dense_dataset_name +'_' + self.sparse_dataset_name + '_' + str(self.seed) +  '_checkpoint.pt')
        self.val_loss_min = val_loss
        
        
def loss_function(args):
    if args.dense_dataset == 'isp' and args.sparse_dataset == 'wb':
        pos_weight = 2
    else:
        pos_weight = 6

    pos_weight = torch.tensor([pos_weight], dtype=torch.float).to(args.gpu)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    return criterion        
        
def compute_metrics(preds, labels, average='macro'):
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()  
    preds_binary = (preds >= 0.5).astype(int)
    precision = precision_score(labels, preds_binary, average=average)
    recall = recall_score(labels, preds_binary, average=average)
    f1 = f1_score(labels, preds_binary, average=average)
    auc = roc_auc_score(labels, preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def save_metrics(metrics, file_path):
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)