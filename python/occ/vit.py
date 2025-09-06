import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from occ.utils import overlay_attn

plt.style.use("ggplot")

logger = logging.getLogger(__name__)

class ViTModel(nn.Module):
    def __init__(self, config):

        n_classes = config['model']['n_classes']
        model_path = config['paths']['model']
        pretrained = config['model']['pretrained_local']

        super(ViTModel, self).__init__()

        if pretrained:
            # load local model checkpoint
            self.model = timm.create_model(config['model']['type'],
                                           num_classes=n_classes,
                                           drop_rate=config['hyperparameters']['dropout_rate'],
                                           attn_drop_rate=config['hyperparameters']['attn_dropout_rate'])
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

            # local models are saved with prefix that needs to be removed
            state_dict = remove_prefix_from_state_dict(state_dict, 'model.')
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            model_dict = self.model.state_dict()

        elif config['model']['pretrained_timm']:
            # load model with pretrained timm weights
            self.model = timm.create_model(config['model']['type'],
                                           num_classes=n_classes,
                                           pretrained=True,
                                           drop_rate=config['hyperparameters']['dropout_rate'],
                                           attn_drop_rate=config['hyperparameters']['attn_dropout_rate'])
        else:
            # load uninitialized model
            self.model = timm.create_model(config['model']['type'],
                                           num_classes=n_classes,
                                           drop_rate=config['hyperparameters']['dropout_rate'],
                                           attn_drop_rate=config['hyperparameters']['attn_dropout_rate'])

        # not supported: option to experiment with freezing some layers of the network (not used in 2025 OCC paper)
        # if config['training']['freeze_layers']:
        #     for name, param in self.model.named_parameters():
        #         if 'fc' not in name:  # 'fc' is typically the final fully connected layer in models like ResNet
        #             param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x

    def train_one_epoch(self, train_loader, criterion, optimizer, classes, device, epoch, config):
        # store data and statistics
        epoch_loss = 0.0
        true_positives = {i: 0 for i in range(len(classes))}
        false_positives = {i: 0 for i in range(len(classes))}
        false_negatives = {i: 0 for i in range(len(classes))}
        precisions = {i: 0 for i in range(len(classes))}
        recalls = {i: 0 for i in range(len(classes))}
        f1s = {i: 0 for i in range(len(classes))}

        self.model.train()
        for i, (data, target) in enumerate(train_loader):
            if device.type == "cuda":
                data, target = data.to(device), target.to(device)

            # train batch
            optimizer.zero_grad()
            output = self.forward(data)
            loss = criterion(output, target)
            loss.backward()

            # calculate F1
            predicted_labels = output.argmax(dim=1)
            for label, pred_label in zip(target, predicted_labels): # this iterates over images in batch
                if label == pred_label:
                    true_positives[int(label)] += 1
                else:
                    false_negatives[int(label)] += 1
                    false_positives[int(pred_label)] += 1

            # update training loss
            epoch_loss += loss.item() # `.item()` for memory leak bug fix

            optimizer.step()

        # if there are no true positives, we need to avoid F1 score calc error
        for i in range(len(classes)):
            if true_positives[i] == 0:
                precisions[i] = 0
                recalls[i] = 0
                f1s[i] = 0
            else:
                precisions[i] = round(true_positives[i] / (true_positives[i] + false_positives[i]), 4)
                recalls[i] = round(true_positives[i] / (true_positives[i] + false_negatives[i]), 4)
                f1s[i] = round(2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]), 4)

        logger.info("Printing classes, precisions, recalls, and F1s respectively...")
        logger.info("\t".join(classes))
        logger.info("\t".join(str(precisions[i]) for i in range(len(classes))))
        logger.info("\t".join(str(recalls[i]) for i in range(len(classes))))
        logger.info("\t".join(str(f1s[i]) for i in range(len(classes))))

        # store epoch statistics for csv output
        summary_results = {
            "model": config['general']['model_id'],
            "epoch": epoch,
            "class": range(15),
            "type": "train",
            "precision": pd.Series(precisions),
            "recall": pd.Series(recalls),
            "f1": pd.Series(f1s),
            "loss": epoch_loss / len(train_loader)
        }
        summary_results = pd.DataFrame(summary_results)

        return (
            epoch_loss / len(train_loader),
            sum(recalls.values()) / len(classes),
            sum(precisions.values()) / len(classes),
            sum(f1s.values()) / len(classes),
            summary_results
            )

    def validate_one_epoch(self, valid_loader, criterion, classes, device, epoch, config, train, valid_f1_max,
                           _print=True, error_analysis=False, attn_maps=[]):
        # store data and statistics
        valid_loss = 0.0
        true_positives = {i: 0 for i in range(len(classes))}
        false_positives = {i: 0 for i in range(len(classes))}
        false_negatives = {i: 0 for i in range(len(classes))}
        precisions = {i: 0 for i in range(len(classes))}
        recalls = {i: 0 for i in range(len(classes))}
        f1s = {i: 0 for i in range(len(classes))}

        # initialize arrays for confusion matrix and dataframe results
        all_targets = []
        all_predictions = []
        all_probs = []
        probs_arrays = []

        # initialize arrays for error analysis
        if error_analysis:
            image_misses = []
            label_misses = []
            preds_misses = []
            probs_misses = []
            attn_misses = []

        self.model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                if device.type == "cuda":
                    if data.is_cuda and target.is_cuda:
                        pass
                    else:
                        data, target = data.to(device), target.to(device)

                # evaluate the epoch
                output = self.model(data)
                loss = criterion(output, target)
                if not train:
                    loss = 0

                # store for error analysis, conf matrix, results, etc
                predicted_labels = output.argmax(dim=1)
                probs = F.softmax(output, dim=1)
                max_probs = probs.max(dim=1).values

                # reformat for confusion matrix and results output
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted_labels.cpu().numpy())
                all_probs.extend(max_probs.cpu().numpy())
                probs_arrays.extend(probs.cpu().numpy())

                for idx, (label, pred_label, prob) in enumerate(zip(target, predicted_labels, max_probs)):
                    # calculate F1
                    if label == pred_label:
                        true_positives[int(label)] += 1
                    else:
                        false_negatives[int(label)] += 1
                        false_positives[int(pred_label)] += 1

                        # store data for error analysis
                        if error_analysis:
                            image_misses.append(data[idx].cpu())
                            label_misses.append(label.cpu().numpy())
                            preds_misses.append(pred_label.cpu().numpy())
                            probs_misses.append(prob.cpu().numpy())
                            # store attention maps for error analysis. they're hooked using code in `main()`
                            if len(attn_maps) > 0 and config['output']['attention_map_overlay']:
                                # append a single map of 12-headed attn for the missed image
                                attn_misses.append(attn_maps[-1][idx])

                # update average validation loss
                if not train:
                    valid_loss += loss
                else:
                    valid_loss += loss.item() # `.item()` for memory leak bug fix

                # save GPU memory
                attn_maps.clear()

                '''
                Error analysis:
                - images are generated of misclassifications
                - includes predicted class, model confidence, and ViT attention map overlay
                '''
                if epoch == config['training']['no_epochs'] and error_analysis:
                    output_dir = config['paths']['errors_dir']
                    os.makedirs(output_dir, exist_ok=True)
                    for idx, (img, label, pred, prob, attn) in enumerate(zip(image_misses, label_misses,
                                                                             preds_misses, probs_misses, attn_misses)):
                        # include model confidence in image
                        prob_str = str(int(round(prob * 1000, 0))).zfill(3)
                        prob = np.round(prob,3)
                        img = img.permute(1, 2, 0).numpy()

                        # write misclassfied images to class-specific folders
                        class_dir = os.path.join(output_dir, classes[label])
                        os.makedirs(class_dir, exist_ok=True)

                        # plot attention heatmap over the image
                        if config['output']['include_attention_overlay']:
                            # take an average over the attention heads (head='agg')
                            attn=overlay_attn(attn, head='agg', 
                                              size=config['image_processing']['resize'], patch_len=config['general']['patch_length'])
                            plt.imshow(img, alpha=0.6)
                            plt.imshow(attn.cpu().numpy(), cmap='jet', alpha=0.4)
                        else:
                            plt.imshow(img)

                        plt.title(f'''Ground Truth: {classes[label]},
                                  Prediction: {classes[pred]}, Confidence: {prob:.1%}''')
                        plt.axis('off')
                        plt.savefig(f"{output_dir}/{classes[label]}/miss_{prob_str}_miss{idx}_true_{classes[label]}_pred_{classes[pred]}.png")
                        plt.close()

                    # remove batch misses
                    image_misses.clear()
                    label_misses.clear()
                    preds_misses.clear()
                    probs_misses.clear()
                    attn_misses.clear()

        # if there are no true positives, we need to avoid F1 score calc error
        for i in range(len(classes)):
            if true_positives[i] == 0:
                precisions[i] = 0
                recalls[i] = 0
                f1s[i] = 0
            else:
                precisions[i] = round(true_positives[i] / (true_positives[i] + false_positives[i]), 4)
                recalls[i] = round(true_positives[i] / (true_positives[i] + false_negatives[i]), 4)
                f1s[i] = round(2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]), 4)
        valid_f1 = sum(f1s.values()) / len(classes)

        if _print:
            logger.info("Printing classes, precisions, recalls, and F1s respectively...")
            logger.info("\t".join(classes))
            logger.info("\t".join(str(precisions[i]) for i in range(len(classes))))
            logger.info("\t".join(str(recalls[i]) for i in range(len(classes))))
            logger.info("\t".join(str(f1s[i]) for i in range(len(classes))))

        # confusion matrix
        if config['output']['confusion_matrix'] & (not train):
            # write only if conditions meet best_f1 or final_epoch, as specified in cfg
            if (
            ((config['output']['confusion_matrix_type'] == 'best_f1') & (valid_f1 > valid_f1_max) ) | 
            ((config['output']['confusion_matrix_type'] == 'final_epoch') & epoch == config['training']['no_epochs'])
            ):
                cm = confusion_matrix(all_targets, all_predictions)
                plt.figure(figsize=(10, 8))
                sn.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                path = os.path.join(config['paths']['conf_mat_output_dir'],
                                    f"confusion_matrix_{config['general']['model_id']}.png")
                plt.savefig(path)
                plt.close()

        results = {
            "targets": all_targets,
            "preds": all_predictions,
            "probs": all_probs,
            "probs_arrays": probs_arrays
        }

        # output for csv
        summary_results = {
            "model": config['general']['model_id'],
            "epoch": epoch,
            "class": range(15),
            "type": "eval",
            "precision": pd.Series(precisions),
            "recall": pd.Series(recalls),
            "f1": pd.Series(f1s),
            "loss": (valid_loss / len(valid_loader))
        }
        summary_results = pd.DataFrame(summary_results)

        return (
            valid_loss / len(valid_loader),
            sum(recalls.values()) / len(classes),
            sum(precisions.values()) / len(classes),
            valid_f1,
            results,
            summary_results
        )

def remove_prefix_from_state_dict(state_dict, prefix):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # Remove the prefix
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
