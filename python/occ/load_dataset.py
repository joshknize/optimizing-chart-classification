import os
import numpy as np
import random

import torch
import torchvision
from torchvision.datasets import DatasetFolder
from torch.utils.data import Subset
    
class LoadDataset:
    def __init__(self, config):
        self.config = config
        
    def load_dataset(config, seed, split):        
        
        data_path = config['paths']['train_folder'] 

        # TODO: PROCESS IMAGES

        # create dataset and dataloader
        torch.manual_seed(seed)
        if split == 'train':
            if config['training']['weighted_data_loader']:
                
                # create probability array for sampling each class (hard-coded to original distribution from custom_estimation_split)
                    # refactor later using os.path and name dicts
                if config['paths']['train_folder'] != 'data/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images':  
                    config_split = {
                        "area": 251/28999,
                        "heatmap": 307/28999,
                        "horizontal_bar": 1129/28999,
                        "horizontal_interval": 480/28999,
                        "line": 11143/28999,
                        "manhattan": 215/28999,
                        "map": 709/28999,
                        "pie": 354/28999,
                        "scatter": 2115/28999,
                        "scatter-line": 2777/28999,
                        "surface": 208/28999,
                        "venn": 174/28999,
                        "vertical_bar": 7366/28999,
                        "vertical_box": 1232/28999,
                        "vertical_interval": 539/28999 
                    }
                else:
                    config_split = {
                        "area": 172/22923,
                        "heatmap": 197/22923,
                        "horizontal_bar": 787/22923,
                        "horizontal_interval": 156/22923,
                        "line": 10556/22923,
                        "manhattan": 176/22923,
                        "map": 533/22923,
                        "pie": 242/22923,
                        "scatter": 1350/22923,
                        "scatter-line": 1818/22923,
                        "surface": 155/22923,
                        "venn": 75/22923,
                        "vertical_bar": 5454/22923,
                        "vertical_box": 763/22923,
                        "vertical_interval": 156/22923 
                    }

                w = config['hyperparameters']['stratified_weight']
                
                # linearly interpolate new sampling distribution of classes
                for key in config_split:
                    config_split[key] = config_split[key] * w + (1/config['model']['n_classes']) * (1 - w)
                
                # assign probability distribution to an array
                stratified_probs = np.zeros(config['model']['n_classes'])
                for i, k in enumerate(config_split):
                    stratified_probs[i] = config_split[k]

                # make probabilities cumulative for sampling strategy
                for i in range(1, len(stratified_probs)):
                    stratified_probs[i] += stratified_probs[i-1]
                
                train_dataset = StratifiedDataset(
                    data_path, 
                    stratified_probs, 
                    total_size = config['hyperparameters']['stratified_size'],
                    data_path = config['paths']['train_folder'],
                    transform=train_transformation
                )    
            else:
                train_dataset = torchvision.datasets.ImageFolder(
                    root=data_path,
                    transform=train_transformation
                )
            
            # data loader doesn't change based on dataset sampling
            data_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config['hyperparameters']['batch_size'],
                num_workers=config['no_workers'],
                shuffle=True
            )

        else:
            val_dataset = torchvision.datasets.ImageFolder(
                root=data_path,
                transform=val_transformation
            )
            if nested:
                val_dataset = Subset(val_dataset, indices)
            data_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size = config['hyperparameters']['batch_size'],
                num_workers=config['no_workers'],
                shuffle=False
            )
        
        return data_loader
    
class StratifiedDataset(DatasetFolder):
    def __init__(self, root, stratified_probs, total_size, data_path, transform=None):
        super().__init__(root, loader=self.pil_loader, extensions=("jpg", "png"))
        
        self.stratified_probs = stratified_probs
        self.total_size = total_size
        self.data_path = data_path
        self.transform = transform

        self.classes = sorted(os.listdir(self.root))
        self.images = self._initialize_images()
        self.samples = self.sample_classes()
          
    @staticmethod
    def pil_loader(path):
        return Image.open(path).convert("RGB")
    
    def _initialize_images(self):
        # create an array of image paths
        images = {}
        for c in self.classes:
            class_path = os.path.join(self.root, c)
            images[c] = [os.path.join(class_path, img) for img in os.listdir(class_path)]

        return images
    
    def sample_classes(self):
        sampled_data = []
        
        while len(sampled_data) < self.total_size:
            # randomly choose class based on stratified probability array
            rnd = random.random()
            for i, p in enumerate(self.stratified_probs):
                if rnd < p:
                    rnd_class = self.classes[i]
                    if len(self.images[rnd_class]) > 0:
                        img = random.choice(self.images[rnd_class])
                        self.images[rnd_class].remove(img)
                        sampled_data.append((img, i))
                    break

        return sampled_data

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)