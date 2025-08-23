import logging
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.datasets import DatasetFolder

from occ.process_images import ProcessImages

logger = logging.getLogger(__name__)

class LoadDataset:
    def __init__(self, config):
        self.config = config

    def load_dataset(config, seed, split):

        if split == 'train':
            data_path = config['paths']['train_folder']
        else:
            data_path = config['paths']['test_folder']

        # obtain train and validation transformation compositions
        processor = ProcessImages(config)
        train_transformation, val_transformation = processor.get_transforms(seed)

        # create dataset and dataloader
        torch.manual_seed(seed)

        if split == 'train':
            # create new sampling distribution for classes
            if config['training']['weighted_data_loader']:

                # create dict of original distribution img counts by class
                config_split = {}
                total_imgs = 0
                class_list = os.listdir(config['paths']['train_folder'])

                # get counts
                for dir in class_list:
                    class_img_count = len(os.listdir(f'{config['paths']['train_folder']}/{dir}'))
                    total_imgs += class_img_count
                    config_split[dir] = class_img_count

                # adjust weights of distributions
                w = config['training']['wdl_weight']
                for dir in config_split:
                    # turn counts into fraction of whole
                    config_split[dir] = config_split[dir] / total_imgs
                    # linearly interpolate new sampling distributions
                    config_split[dir] = config_split[dir] * w + (1/config['model']['n_classes']) * (1 - w)

                # assign probability distribution to an array
                sampling_distribution = np.zeros(config['model']['n_classes'])
                for i, k in enumerate(config_split):
                    sampling_distribution[i] = config_split[k]

                # make probabilities cumulative (for sampling logic later)
                for i in range(1, len(sampling_distribution)):
                    sampling_distribution[i] += sampling_distribution[i-1]

                train_dataset = WeightedSubsample(
                    data_path,
                    sampling_distribution,
                    total_size = config['training']['wdl_size'],
                    data_path = config['paths']['train_folder'],
                    transform=train_transformation
                )
            # otherwise, load dataset from original distribution
            else:
                train_dataset = torchvision.datasets.ImageFolder(
                    root=data_path,
                    transform=train_transformation
                )

            data_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config['hyperparameters']['batch_size'],
                num_workers=config['general']['no_workers'],
                shuffle=True
            )

        else:
            val_dataset = torchvision.datasets.ImageFolder(
                root=data_path,
                transform=val_transformation
            )
            data_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size = config['hyperparameters']['batch_size'],
                num_workers=config['general']['no_workers'],
                shuffle=False
            )

        return data_loader

class WeightedSubsample(DatasetFolder):
    def __init__(self, root, sampling_distribution, total_size, data_path, transform=None):
        super().__init__(root, loader=self.pil_loader, extensions=("jpg", "png"))

        self.sampling_distribution = sampling_distribution
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
            # randomly choose class based on weighted dataloader probability array
            rnd = random.random()
            for i, p in enumerate(self.sampling_distribution):
                # array is cumulative probability distribution
                if rnd < p:
                    rnd_class = self.classes[i]
                    # sampling without replacement; select new class if directory is empty
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
