import gc
import logging
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from kornia.losses import FocalLoss

from occ.load_dataset import LoadDataset

logger = logging.getLogger(__name__)


def run_model(model, config, device, attn_maps=[], verbose=True):
    """
    Wrapper function for training and validation loop.

    Args:
        model (torch.nn.Module)
        config (dict): dictionary from json config file
        device (str): 'cpu' or 'cuda'
        attn_maps (list, optional): Empty list for attention maps if error analysis enabled.
        verbose (bool, optional)

    Returns:
        `results` dictionary
    """

    torch.set_default_dtype(torch.float32)
    a = _run(model, config, device, attn_maps=attn_maps, verbose=verbose)

    return a

def _run(model, config, device, attn_maps=[], verbose=True):
    """
    Run the training and validation loop.

    Args:
        (as above)

    Returns:
        `results` dictionary
    """

    # set the loss function. configurable in the json config file
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

    # use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['hyperparameters']['learning_rate'])

    if verbose:
        logger.info(f"INITIALIZING TRAINING ON {device}")
        start_time = datetime.now()
        logger.info(f"Start Time: {start_time}")

    logs = fit(
        model,
        device,
        criterion,
        optimizer,
        config,
        attn_maps,
        verbose
    )

    if verbose:
        logger.info(f"Execution time: {datetime.now() - start_time}")

    return logs['results']

def fit(model, device, criterion, optimizer, config, attn_maps=[], verbose=True):
    """
    Function that runs training loop, store statistics, saves model, and prints results.

    Args:
        (as above)

    Returns:
        `results` dictionary
    """

    # track change in validation loss
    valid_f1_max = 0
    train_f1_max = 0

    # keep track of losses
    train_losses = []
    train_f1s = []
    valid_losses = []
    valid_f1s = []

    # store results for csv output / data analysis
    summary_df = pd.DataFrame()
    summary_results_list = []

    classes = sorted(os.listdir(config['paths']['train_folder']))

    train = config['general']['train_model'] # boolean value

    # train model one epoch at a time
    for epoch in range(config['training']['start_epoch'], config['training']['no_epochs'] + 1):

        if train:
            train_loader = LoadDataset.load_dataset(
                config,
                # use seed+epoch to set a new seed for each epoch and keep results reproducible
                config['general']['seed']+epoch,
                split='train'
            )
        valid_loader = LoadDataset.load_dataset(
            config,
            config['general']['seed']+epoch,
            split='test'
        )

        # training and validating each epoch is a separate function call each time
        if train:
            logger.info(f"{'='*50}")
            logger.info(f"EPOCH {epoch} - TRAINING...")
            train_loss, train_recall, train_precision, train_f1, summary_results = model.train_one_epoch(
                train_loader, criterion, optimizer, classes, device, epoch, config
                )
            if verbose:
                logger.info(
                    f"\n\t[TRAIN] EPOCH {epoch} - F1/Recall/Precision/Loss: "
                    f"{train_f1}\t{train_recall}\t{train_precision}\t{train_loss}\n"
                )
            # store results for returning results dict to main()
            train_losses.append(train_loss)
            train_f1s.append(train_f1)
            # append results for csv output at end of run
            summary_results_list.append(summary_results)

        if valid_loader is not None:
            if verbose:
                logger.info(f"EPOCH {epoch} - VALIDATING...")
            valid_loss, valid_recall, valid_precision, valid_f1, results, summary_results = model.validate_one_epoch(
                valid_loader, criterion, classes, device, epoch, config, train, valid_f1_max,
                verbose, error_analysis=config['output']['error_analysis'], attn_maps=attn_maps
            )
            if verbose:
                logger.info(
                    f"\t[VALID] F1/Recall/Precision/Loss: "
                    f"{valid_f1}\t{valid_recall}\t{valid_precision}\t{valid_loss}\n"
                )
            # store results for returning results dict to main()
            valid_losses.append(valid_loss)
            valid_f1s.append(valid_f1)
            # append results for csv output at end of run
            summary_results_list.append(summary_results)

            # save models at various states depending on config file
            path = os.path.join(config['paths']['model_save_dir'],
                                f"model_{config['general']['model_id']}_{epoch}.pth")
            # save model state at last epoch
            if epoch == config['hyperparameters']['no_epochs']:
                torch.save(model.state_dict(), path)
                logger.info(f"Model saved at epoch {epoch} -- Training complete")
            # save model state at checkpoints
            elif epoch % config['training']['chkpt_freq'] == 0:
                torch.save(model.state_dict(), path)
                logger.info(f"Model saved at epoch {epoch} -- Checkpoint")
            # save model if training score improves
            elif train_f1 > train_f1_max:
                path = os.path.join(config['paths']['model_save_dir'],
                                    f"model_{config['general']['model_id']}_best-train.pth")
                torch.save(model.state_dict(), path)
                logger.info(f"Model saved at epoch {epoch} -- Training score improved")
                train_f1_max = train
            # save model if validation score improves
            elif valid_f1 > valid_f1_max:
                path = os.path.join(config['paths']['model_save_dir'],
                                    f"model_{config['general']['model_id']}_best-validation.pth")
                torch.save(model.state_dict(), path)
                logger.info(f"Model saved at epoch {epoch} -- Validation score improved")
                valid_f1_max = valid_f1

        # ameliorate memory issues
        gc.collect()

    # concatenate all results at once, write to csv
    summary_df = pd.concat(summary_results_list, axis=0)
    path = os.path.join(config['paths']['csv_output'], f"{config['general']['model_id']}.csv")
    summary_df.to_csv(path, index=False)

    if verbose:
        logger.info("Printing valid losses, F1s respectively...")
        epoch_count = config['training']['no_epochs'] - config['training']['start_epoch'] + 1
        if train:
            logger.info("\t".join(str(train_losses[i]) for i in range(epoch_count)))
        if valid_loss != 0:
            logger.info("\t".join(str(valid_losses[i]) for i in range(epoch_count)))
        logger.info("\t".join(str(valid_f1s[i]) for i in range(epoch_count)))

    # return results dict
    return {
        "train_losses": train_losses,
        "train_f1": train_f1s,
        "valid_losses": valid_losses,
        "valid_f1": valid_f1s,
        "results": results
    }
