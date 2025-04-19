"""
ViTModel (class)
__init__: configure model architecture, weights, and layers
forward:
train_one_epoch: iterates over batches, then images in batch, to train model and calculate scores/metrics
validate_one_epoch: iterates over batches/images to calculate metrics and return confusion matrix

"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import timm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from occ.utils import overlay_attn

def remove_prefix_from_state_dict(state_dict, prefix):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # Remove the prefix
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

class ViTModel(nn.Module):
    def __init__(self, config): # pretrained from local model 
        
        n_classes = config['model']['n_classes']
        model_path = config['paths']['model'] 
        pretrained = config['model']['pretrained_local']

        super(ViTModel, self).__init__() # allows you to call methods from base nn.Module class
        
        if pretrained:
            # load local model checkpoint
            self.model = timm.create_model(config['model']['type'], 
                                           num_classes=n_classes,
                                           drop_rate=config['hyperparameters']['drop_rate'],
                                           attn_drop_rate=config['hyperparameters']['attn_drop_rate'])
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
                                           drop_rate=config['hyperparameters']['drop_rate'],
                                           attn_drop_rate=config['hyperparameters']['attn_drop_rate'])
        else:
            # load uninitialized model
            self.model = timm.create_model(config['model']['type'], 
                                           num_classes=n_classes,
                                           drop_rate=config['hyperparameters']['drop_rate'],
                                           attn_drop_rate=config['hyperparameters']['attn_drop_rate'])
            
        if config['hyperparameters']['freeze_layers']:
            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # 'fc' is typically the final fully connected layer in models like ResNet
                    param.requires_grad = False
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def train_one_epoch(self, train_loader, criterion, optimizer, classes, device, epoch, config):
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
    
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.forward(data) ## looks like a tensor with shape [batch size, classes] i.e. a probability for each image/class combination
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward() 
            # Calculate F1
            predicted_labels = output.argmax(dim=1)
            for label, pred_label in zip(target, predicted_labels): # this iterates over images in batch
                if label == pred_label:
                    true_positives[int(label)] += 1
                else:
                    false_negatives[int(label)] += 1
                    false_positives[int(pred_label)] += 1
                    
            # update training loss, f1
            epoch_loss += loss.item() # memory leak bug fix

            optimizer.step()
        
        for i in range(len(classes)): # manual adjustment for cases where no true positives happen to avoid F1 score calc error
            if true_positives[i] == 0:
                precisions[i] = 0
                recalls[i] = 0
                f1s[i] = 0
            else:
                precisions[i] = round(true_positives[i] / (true_positives[i] + false_positives[i]), 4)
                recalls[i] = round(true_positives[i] / (true_positives[i] + false_negatives[i]), 4)
                f1s[i] = round(2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]), 4)
        
        print("Printing classes, precisions, recalls, and F1s respectively...")
        print("\t".join(classes))
        print("\t".join(str(precisions[i]) for i in range(len(classes))))
        print("\t".join(str(recalls[i]) for i in range(len(classes))))
        print("\t".join(str(f1s[i]) for i in range(len(classes))))

        precisions2 = pd.Series(precisions)
        recalls2 = pd.Series(recalls)
        f1s2 = pd.Series(f1s)
        summary_results = {
            "model": config['model_id'],
            "epoch": epoch,
            "class": range(15),
            "type": "train",
            "precision": precisions2,
            "recall": recalls2,
            "f1": f1s2,
            "loss": epoch_loss / len(train_loader)
        }
        summary_results = pd.DataFrame(summary_results)


        return epoch_loss / len(train_loader), sum(recalls.values()) / len(classes), sum(precisions.values()) / len(classes), sum(f1s.values()) / len(classes), summary_results

    def validate_one_epoch(self, valid_loader, criterion, classes, device, epoch, config, train, valid_f1_max,
                           _print=True, error_analysis='none', attn_maps=[]):
        valid_loss = 0.0
        true_positives = {i: 0 for i in range(len(classes))}
        false_positives = {i: 0 for i in range(len(classes))}
        false_negatives = {i: 0 for i in range(len(classes))}
        precisions = {i: 0 for i in range(len(classes))}
        recalls = {i: 0 for i in range(len(classes))}
        f1s = {i: 0 for i in range(len(classes))}
        
        all_targets = []
        all_predictions = []
        all_probs = []
        probs_arrays = []
        
        if error_analysis != 'none':
            image_misses = []
            label_misses = []
            preds_misses = []
            probs_misses = []
            attn_misses = []

        self.model.eval() # puts model in evaluation mode
        
        with torch.no_grad():
            for data, target in valid_loader:
                # move tensors to GPU if CUDA is available
                    # unsure if data is already on CUDA
                if device.type == "cuda":
                    if data.is_cuda and target.is_cuda:
                        pass
                    else:
                        data, target = data.to(device), target.to(device)

                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                # calculate the batch loss
                loss = criterion(output, target) 
                if train==False:
                    loss = 0
                # Calculate F1
                predicted_labels = output.argmax(dim=1)
                probs = F.softmax(output, dim=1)
                max_probs = probs.max(dim=1).values
                
                if not train:
                    all_targets.extend(target.cpu().numpy())
                    all_predictions.extend(predicted_labels.cpu().numpy())
                    all_probs.extend(max_probs.cpu().numpy())
                    probs_arrays.extend(probs.cpu().numpy())
                
                for idx, (label, pred_label, prob) in enumerate(zip(target, predicted_labels, max_probs)):
                    if label == pred_label:
                        true_positives[int(label)] += 1

                        if error_analysis == 'all_attention_overlay' and (label == 7 or label == 8):
                            image_misses.append(data[idx].cpu())
                            label_misses.append(label.cpu().numpy())
                            preds_misses.append(pred_label.cpu().numpy())
                            probs_misses.append(prob.cpu().numpy())
                            if len(attn_maps) > 0:
                                attn_misses.append(attn_maps[-1][idx]) # append a single map of 12-headed attn for the missed image

                    else:
                        false_negatives[int(label)] += 1
                        false_positives[int(pred_label)] += 1
                        
                        if error_analysis != 'none' and label>4:
                            image_misses.append(data[idx].cpu())
                            label_misses.append(label.cpu().numpy())
                            preds_misses.append(pred_label.cpu().numpy())
                            probs_misses.append(prob.cpu().numpy())
                            if error_analysis in ['attention_overlay', 'all_attention_overlay'] and (len(attn_maps) > 0):
                                attn_misses.append(attn_maps[-1][idx]) # append a single map of 12-headed attn for the missed image
                                
                # update average validation loss
                if not train:
                    valid_loss += loss
                else:
                    valid_loss += loss.item() # memory leak bug fix
                
                # save GPU memory
                attn_maps.clear()
                
                # move error analysis within loop to save GPU memory 
                if epoch == config['hyperparameters']['no_epochs'] and error_analysis != 'none':
                    output_dir = config['evaluation']['errors_dir']
                    os.makedirs(output_dir, exist_ok=True)
                    for idx, (img, label, pred, prob, attn) in enumerate(zip(image_misses, label_misses, preds_misses, probs_misses, attn_misses)):
                        prob_str = str(int(round(prob * 1000, 0))).zfill(3)
                        prob = np.round(prob,3)
                        img = img.permute(1, 2, 0).numpy()
                        
                        class_dir = os.path.join(output_dir, classes[label])
                        os.makedirs(class_dir, exist_ok=True)
                        
                        if error_analysis in ['attention_overlay', 'all_attention_overlay']:
                            attn=overlay_attn(attn)
                            plt.imshow(img, alpha=0.6)
                            plt.imshow(attn.cpu().numpy(), cmap='jet', alpha=0.4)
                        else:
                            plt.imshow(img)
                        
                        plt.title(f'Ground Truth: {classes[label]}, Prediction: {classes[pred]}, Confidence: {prob:.1%}')
                        plt.axis('off')
                            
                        if classes[label] != classes[pred]:
                            cat = 'miss'
                        else:
                            cat = 'correct'

                        plt.savefig(f"{output_dir}/{classes[label]}/{cat}_{prob_str}_miss{idx}_true_{classes[label]}_pred_{classes[pred]}.png")
                        plt.close()
                        
                    # remove batch misses        
                    image_misses.clear()
                    label_misses.clear()
                    preds_misses.clear()
                    probs_misses.clear()
                    attn_misses.clear()
                
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
            print("Printing classes, precisions, recalls, and F1s respectively...")
            print("\t".join(classes))
            print("\t".join(str(precisions[i]) for i in range(len(classes))))
            print("\t".join(str(recalls[i]) for i in range(len(classes))))
            print("\t".join(str(f1s[i]) for i in range(len(classes))))
            
        # Confusion matrix: update 1/15 - only overwrite conf mat if there's a new best F1 score
        if config['evaluation']['confusion_matrix'] & (valid_f1 > valid_f1_max) & (not train):    
            cm = confusion_matrix(all_targets, all_predictions)
            plt.figure(figsize=(10, 8))
            sn.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
            plt.xlabel('Predicted') # (57 area images)
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            path = os.path.join(config['evaluation']['conf_mat_output_dir'], f"confusion_matrix_{config['model_id']}.png")
            plt.savefig(path)
            plt.close()
                    
        results = {
            "targets": all_targets,
            "preds": all_predictions,
            "probs": all_probs,
            "probs_arrays": probs_arrays
        }
        
        precisions2 = pd.Series(precisions)
        recalls2 = pd.Series(recalls)
        f1s2 = pd.Series(f1s)
        summary_results = {
            "model": config['model_id'],
            "epoch": epoch,
            "class": range(15),
            "type": "eval",
            "precision": precisions2,
            "recall": recalls2,
            "f1": f1s2,
            "loss": (valid_loss / len(valid_loader))
        }
        summary_results = pd.DataFrame(summary_results)
        
        return 0 if valid_loss == 0 else valid_loss / len(valid_loader), sum(recalls.values()) / len(classes), sum(precisions.values()) / len(classes), valid_f1, results, summary_results
    