import os
import pandas as pd

import torch
import torch.nn as nn

import gc
from datetime import datetime

from kornia.losses import FocalLoss

from occ.load_dataset import LoadDataset

def run_model(model, config, device, attn_maps=[], verbose=True):
        torch.set_default_dtype(torch.float32)
        a = _run(model, config, device, attn_maps=attn_maps, verbose=verbose)
        return a

def _run(model, config, device, attn_maps=[], verbose=True):
    
    loss_function_map = {
        "CrossEntropyLoss": nn.CrossEntropyLoss(),
        "FocalLoss": FocalLoss(
                    alpha=config['loss']['alpha'],
                    gamma=config['loss']['gamma'],
                    reduction=config['loss']['reduction']),
        "CEL_LabelSmooth": nn.CrossEntropyLoss(label_smoothing=config['loss']['label_smoothing'])
    }

    criterion = loss_function_map[config['loss']['name']]
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['hyperparameters']['learning_rate'])
    
    if verbose:
        print(f"INITIALIZING TRAINING ON {device}")
        start_time = datetime.now()
        print(f"Start Time: {start_time}")
    
    logs = fit_cpu(
        model,
        device,
        criterion,
        optimizer,
        config,
        attn_maps,
        verbose
    )

    if verbose:
        print(f"Execution time: {datetime.now() - start_time}")
    
    return logs['results']

def fit_cpu(model, device, criterion, optimizer, config, attn_maps=[], verbose=True):

    # track change in validation loss
    valid_loss_min = float('inf') 
    valid_f1_max = 0
    train_f1_max = 0

    # keeping track of losses
    train_losses = []
    train_recalls = []
    train_precisions = []
    train_f1s = []
    valid_losses = []
    valid_recalls = []
    valid_precisions = []
    valid_f1s = []

    # store results for csv output / data analysis
    results_list = {}
    summary_df = pd.DataFrame()
    summary_results_list = []
    
    classes = sorted(os.listdir(config['paths']['train_folder']))

    train = config['general']['train_model']
    
    for epoch in range(config['training']['start_epoch'], config['training']['no_epochs'] + 1):
        
        if train:
            train_loader = LoadDataset.load_dataset(
                config, 
                config['seed']+epoch, 
                split = 'train'
            ) 
        valid_loader = LoadDataset.load_dataset(
            config['paths']['test_folder'], 
            config, 
            config['seed']+epoch, 
            split = 'test'
        ) 
    
        if train:
            # always print epoch number
            print(f"{'='*50}")
            print(f"EPOCH {epoch} - TRAINING...")
            train_loss, train_recall, train_precision, train_f1, summary_results = model.train_one_epoch(train_loader, criterion, optimizer, classes, device, epoch, config)
            if verbose:
                print(f"\n\t[TRAIN] EPOCH {epoch} - F1/Recall/Precision/Loss: {train_f1}\t{train_recall}\t{train_precision}\t{train_loss}\n")
            train_losses.append(train_loss)
            train_f1s.append(train_f1)
            summary_results_list.append(summary_results)

        if valid_loader is not None:
            if verbose:
                print(f"EPOCH {epoch} - VALIDATING...")
            valid_loss, valid_recall, valid_precision, valid_f1, results, summary_results = model.validate_one_epoch(
                valid_loader, criterion, classes, device, epoch, config, train, valid_f1_max,
                verbose, error_analysis=config['evaluation']['error_analysis'], attn_maps=attn_maps
            )
            if not train:
                results_string = f'model{epoch}'
                results_list[results_string] = results
            if verbose:
                print(f"\t[VALID] F1/Recall/Precision/Loss: {valid_f1}\t{valid_recall}\t{valid_precision}\t{valid_loss}\n")
            valid_losses.append(valid_loss)
            valid_f1s.append(valid_f1)
            summary_results_list.append(summary_results)
            
            # save model if validation f1 has increased
            if (valid_f1 > valid_f1_max) and train==True and verbose==True and not config['fixed_epochs']:
                print(
                    "Validation f1 increased ({:.4f} --> {:.4f}).  Saving model ...".format(
                        valid_f1_max, valid_f1
                    )
                )
                path = os.path.join(config['paths']['model_save_dir'], f"model_{config['model_id']}.pth")
                torch.save(model.state_dict(), path)

                valid_f1_max = valid_f1
            print(f"max valid F1 score = {valid_f1_max}")

            if config['fixed_epochs']:
                path = os.path.join(config['paths']['model_save_dir'], f"model_{config['model_id']}_{epoch}.pth")
                if epoch == config['hyperparameters']['no_epochs']:
                    torch.save(model.state_dict(), path)
                    print(f"Model saved at epoch {epoch}")
                elif epoch % config['fixed_epochs_chkpt'] == 0:
                    torch.save(model.state_dict(), path)
                    print(f"Model saved at epoch {epoch}")
                elif train_f1 > train_f1_max:
                    path = os.path.join(config['paths']['model_save_dir'], f"model_{config['model_id']}_best-train.pth")
                    torch.save(model.state_dict(), path)
                    print(f"Model saved at epoch {epoch}")
                    train_f1_max = train

        # ameliorate memory issues
        gc.collect()

    # concatenate all results at once
    summary_df = pd.concat(summary_results_list, axis=0)
    path = os.path.join(config['paths']['csv_output'], f"{config['model_id']}.csv")
    summary_df.to_csv(path, index=False)   
    
    if verbose:
        print("Printing valid losses, F1s respectively...")
        epoch_count = epochs - config['hyperparameters']['start_epoch'] + 1
        if train:
            print("\t".join(str(train_losses[i]) for i in range(epoch_count)))
        if valid_loss != 0:
            print("\t".join(str(valid_losses[i]) for i in range(epoch_count)))
        print("\t".join(str(valid_f1s[i]) for i in range(epoch_count)))

    return {
        "valid_losses": valid_losses,
        "valid_f1": valid_f1s,
        "results": results
    }