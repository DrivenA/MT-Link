import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
from utils import EarlyStopping
import random
from data_loader import *
from model import *
from utils import *
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
l1_lambda = 1e-4
l2_lambda = 1e-4

def train_model(train_samples, val_samples, model, optimizer, devices, args, logger, seed):
    BATCH_SIZE = args.batch_size
    PRINT_INTERVAL = 1 
    criterion = loss_function(args)

    avg_train_losses, avg_valid_losses = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    train_aucs, val_aucs = [], []
    
    early_stopping = EarlyStopping(logger=logger, sparse_dataset_name=args.sparse_dataset,
                                   dense_dataset_name=args.dense_dataset, seed = seed, patience=args.patience, verbose=True, save = args.save)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9)
    
    for epoch_idx in tqdm(range(args.epochs), desc="EPOCH"):
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        model.train()
        training_len = len(train_samples)
        train_losses_epoch = []
        all_train_preds = []
        all_train_labels = []

        for batch_idx in tqdm(range(0, training_len, BATCH_SIZE), desc="Train_index"):
            if batch_idx + BATCH_SIZE > training_len:
                continue
            samples = copy.deepcopy(train_samples[batch_idx:batch_idx + BATCH_SIZE])
            batch_data = create_batch(samples, devices)

            for _ in range(2):
                optimizer.zero_grad()
                dense_loc_tensor, dense_tim_tensor, dense_tims_tensor, dense_len_tensor, sparse_loc_tensor, sparse_tim_tensor, sparse_tims_tensor, sparse_len_tensor, labels = batch_data
                
                with autocast():
                    scores = model(dense_loc_tensor, dense_tim_tensor, dense_tims_tensor, dense_len_tensor,
                                sparse_loc_tensor, sparse_tim_tensor, sparse_tims_tensor, sparse_len_tensor)
                    
                    dense_loc_tensor, dense_tim_tensor, dense_tims_tensor, dense_len_tensor, sparse_loc_tensor, sparse_tim_tensor, sparse_tims_tensor, sparse_len_tensor = sparse_loc_tensor, sparse_tim_tensor, sparse_tims_tensor, sparse_len_tensor, dense_loc_tensor, dense_tim_tensor, dense_tims_tensor, dense_len_tensor

                    loss = criterion(scores, labels.unsqueeze(1).float())
                    l1_reg = 0
                    for param in model.parameters():
                        l1_reg += torch.sum(torch.abs(param))
                    l2_reg = 0
                    for param in model.parameters():
                        l2_reg += torch.sum(torch.pow(param, 2))
                    loss = loss + l1_lambda * l1_reg + l2_lambda * l2_reg

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses_epoch.append(loss.item())
                all_train_preds.append(scores)
                all_train_labels.append(labels)

        avg_train_loss = sum(train_losses_epoch) / len(train_losses_epoch)
        avg_train_losses.append(avg_train_loss)
        all_train_preds = torch.cat(all_train_preds)
        all_train_labels = torch.cat(all_train_labels)
        train_metrics = compute_metrics(all_train_preds, all_train_labels)
        train_precisions.append(train_metrics['precision'])
        train_recalls.append(train_metrics['recall'])
        train_f1s.append(train_metrics['f1'])
        train_aucs.append(train_metrics['auc'])

        model.eval()
        valid_losses = []
        all_valid_preds = []
        all_valid_labels = []
        with torch.no_grad():
            validation_len = len(val_samples)
            for batch_idx in tqdm(range(0, validation_len, BATCH_SIZE), desc="Validation_index"):
                
                if batch_idx + BATCH_SIZE > validation_len:
                    continue
                samples = copy.deepcopy(val_samples[batch_idx:batch_idx + BATCH_SIZE])
                batch_data = create_batch(samples, devices)
                dense_loc_tensor, dense_tim_tensor, dense_tims_tensor, dense_len_tensor, sparse_loc_tensor, sparse_tim_tensor, sparse_tims_tensor, sparse_len_tensor, labels = batch_data
                
                with autocast():
                    scores = model(dense_loc_tensor, dense_tim_tensor, dense_tims_tensor, dense_len_tensor,
                                sparse_loc_tensor, sparse_tim_tensor, sparse_tims_tensor, sparse_len_tensor)
                    
                    loss = criterion(scores, labels.unsqueeze(1).float())

                valid_losses.append(loss.item())
                all_valid_preds.append(scores)
                all_valid_labels.append(labels)

        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        avg_valid_losses.append(avg_valid_loss)
        all_valid_preds = torch.cat(all_valid_preds)
        all_valid_labels = torch.cat(all_valid_labels)
        valid_metrics = compute_metrics(all_valid_preds, all_valid_labels)
        val_precisions.append(valid_metrics['precision'])
        val_recalls.append(valid_metrics['recall'])
        val_f1s.append(valid_metrics['f1'])
        val_aucs.append(valid_metrics['auc'])

        if (epoch_idx + 1) % PRINT_INTERVAL == 0 or epoch_idx == args.epochs - 1:
            logger.info(f"Epoch [{epoch_idx + 1}/{args.epochs}], \n"
                        f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, \n"
                        f"Valid Precision: {valid_metrics['precision']:.4f}, Valid Recall: {valid_metrics['recall']:.4f}, Valid F1: {valid_metrics['f1']:.4f}, \n"
                        f"Valid AUC: {valid_metrics['auc']:.4f}\n")

        early_stopping(avg_valid_losses[-1], model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
        scheduler.step()

    save_metrics({
        'train_losses': avg_train_losses,
        'valid_losses': avg_valid_losses
    }, args.save + 'train_metrics.json')



def test_model(test_samples, model, devices, args, logger):
    METRIC = 'macro'
    BATCH_SIZE = args.batch_size
    criterion = loss_function(args)
    model.eval()
    test_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        testing_len = len(test_samples)
        for batch_idx in tqdm(range(0, testing_len, BATCH_SIZE), desc="Test_index (●'◡'●)"):
            if batch_idx + BATCH_SIZE > testing_len:
                continue

            samples = copy.deepcopy(test_samples[batch_idx:batch_idx + BATCH_SIZE])
            batch_data = create_batch(samples, devices)
            dense_loc_tensor, dense_tim_tensor, dense_tims_tensor, dense_len_tensor, sparse_loc_tensor, sparse_tim_tensor, sparse_tims_tensor, sparse_len_tensor, labels = batch_data
            scores = model(dense_loc_tensor, dense_tim_tensor, dense_tims_tensor, dense_len_tensor,
                                sparse_loc_tensor, sparse_tim_tensor, sparse_tims_tensor, sparse_len_tensor)
            loss = criterion(scores, labels.unsqueeze(1).float())
            test_losses.append(loss.item())
            all_preds.append(scores)
            all_labels.append(labels)

    avg_test_loss = sum(test_losses) / len(test_losses)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    logger.info(f"Test Loss: {avg_test_loss:.4f}, \n"
                f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}\n")
    args_dict = vars(args)
    save_metrics({
        'args_settings': args_dict,
        'test_loss': avg_test_loss,
        'test_precision': metrics['precision'],
        'test_recall': metrics['recall'],
        'test_f1': metrics['f1'],
        'test_auc': metrics['auc']
    }, args.save + 'test_metrics.json')

    return metrics
