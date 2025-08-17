import os
import random
import string
from itertools import product

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageDraw, ImageFont


def systematic_crop(img, epoch, resize):
    '''
    Crop images by four corners and then center. Only works for epoch evaluation <= 5
    '''
    
    width, height = img.size
    
    if width < resize or height < resize:
        padding = (
            max(0, resize - height),
            max(0, resize - height))
        img = transforms.functional.pad(img, padding, fill=(255,255,255))
        
        width, height = img.size
        
    if epoch == 1:
        top, left = 0, 0
    elif epoch == 2:
        top, left = 0, width-resize
    elif epoch == 3:
        top, left = height-resize, width-resize
    elif epoch == 4:
        top, left = height-resize, 0
    elif epoch == 5:
        top = (height - resize) / 2
        left = (width - resize) / 2
    else:
        raise ValueError("Epoch out of systematic crop allowed range")
    
    return transforms.functional.crop(img, top, left, height=resize, width=resize)

class ResizingHelper:
    def __init__(self, size):
        self.__size = size

    def resize_on_smaller_dim(self, img):
        # calculate the new size of the smallest dimension
        scale = round(self.__size[0] * min(img.size) / max(img.size))

        return transforms.functional.resize(img, size=scale)

    def pad_image(self, img):
        padding = (
            (self.__size[0] - img.size[0]) // 2,  # Left padding
            (self.__size[0] - img.size[1]) // 2,  # Top padding
            (self.__size[0] - img.size[0] + 1) // 2,  # Right padding
            (self.__size[0] - img.size[1] + 1) // 2  # Bottom padding
        )

        return transforms.functional.pad(img,padding=padding,fill=(255, 255, 255))

class LoadDataset:
    def __init__(self, config):
        self.config = config
        
    def load_dataset(data_path, config, seed, split, indices=None, nested=False):        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
        # store augment parameters in smaller dictionaries for easier access
        augments = config['hyperparameters']['augmentation']
        annotation_augments = config['hyperparameters']['annotations']
        
        epoch = seed - config['seed']
        
        # create transformation / augmentation pipeline
        size = (config['image']['resize'],config['image']['resize'])

        # KD: added to remove the dependencies on Lambda functions
        resizer_helper = ResizingHelper(size)

        if config['image']['method'] == 'crop':
            transforms_size = [
                transforms.RandomCrop(size, pad_if_needed=True, fill = 255)
            ]
        elif config['image']['method'] == 'resize':    
            transforms_size = [
                transforms.Resize(size)
            ]
        elif config['image']['method'] == 'resize_keep_aspect_ratio':
            # transforms.Resize scales smallest dim to `size` arg
            transforms_size = [
                transforms.Lambda(resizer_helper.resize_on_smaller_dim),
                # pad with white to ViT input size
                    # possible inefficiency; might break batch processing
                transforms.Lambda(resizer_helper.pad_image)
            ]
        elif config['image']['method'] == 'prop_crop':
            transforms_size = [
                transforms.Lambda(lambda img: transforms.functional.crop(
                    img,
                    top=random.randint(0, int(img.size[1] * 0.5)),
                    left=random.randint(0, int(img.size[0] * 0.5)),
                    height=int(img.size[1] * 0.5),
                    width=int(img.size[0] * 0.5))),
                transforms.Resize(size)
            ]
        elif config['image']['method'] == 'systematic_crop':
            resize = config['image']['resize']
            transforms_size = [
                transforms.Lambda(lambda img: systematic_crop(img, epoch, resize))    
            ]
        
        brightness_jitter = transforms.ColorJitter(brightness=augments['brightness'])
        contrast_jitter = transforms.ColorJitter(contrast=augments['contrast'])
        saturation_jitter = transforms.ColorJitter(saturation=augments['saturation'])
        hue_jitter = transforms.ColorJitter(hue=augments['hue']) 

        # apply each jitter augment at a specified probability to increase augmentation combinations
        jitter_probs = config['hyperparameters']['augmentation']['jitter_probs']
        if len(jitter_probs) == 1:
            jitter_probs = jitter_probs * 4
        
        jitter_transform = transforms.Compose([
            transforms.RandomApply([brightness_jitter], p=jitter_probs[0]),
            transforms.RandomApply([contrast_jitter], p=jitter_probs[1]),
            transforms.RandomApply([saturation_jitter], p=jitter_probs[2]),
            transforms.RandomApply([hue_jitter], p=jitter_probs[3])
        ])

        transforms_augments = [
            AnnotationAugment(annotation_augments),
            transforms.RandomHorizontalFlip(augments['hor_flip']),
            transforms.RandomInvert(augments['invert']),
            jitter_transform,            
            GridAugment(config['hyperparameters']['grid']['prob'], config['hyperparameters']['grid']['col'],
                        config['hyperparameters']['grid']['lwd'], config['hyperparameters']['grid']['pct_grid'], config['hyperparameters']['grid']['uniform']),
            # mask image by probability, patch or "block", and color
            ApplyMasks(config['hyperparameters']['mask']['prob'],
                       config['hyperparameters']['mask']['mask_length'], 
                       config['hyperparameters']['mask']['num_patches'], 
                       config['hyperparameters']['mask']['pixel_value'],
                       config['hyperparameters']['mask']['by_patch'],
                       config['hyperparameters']['patch_length'],
                       config['hyperparameters']['mask']['pct_mask']),
        ]
            
        transforms_general = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        
        train_transformation = transforms.Compose(
            transforms_size + transforms_augments + transforms_general
        )
        val_transformation = transforms.Compose(
            transforms_size + transforms_general

        )      
    
class ApplyMasks:
    '''
    Given an input number, color, and size, this class masks a random section of each training image with square patches.
    '''
    def __init__(self, prob: float, mask_len: int, num_patches: int, color: int, by_patch: bool, patch_len: int, pct_mask: float):
        self.prob = prob
        self.mask_len = mask_len
        self.num_patches = num_patches # mask patches
        self.color = color
        self.by_patch = by_patch
        self.patch_len = patch_len
        self.pct_mask = pct_mask

    def __call__(self, img):       

        # return image if no mask settings are configured
        if (self.num_patches < 1) and (not self.by_patch):
            return img

        # return image probabilistically
        if random.random() > self.prob:
            return img

        # convert to np.array
        img = np.array(img)

        h, w, _ = img.shape

        if not self.by_patch:
            # select random cut of img and replace pixels in all channels with pixel value
            for _ in range(self.num_patches):
                top_left_x = random.randint(0, max(0, w - self.mask_len))
                top_left_y = random.randint(0, max(0, h - self.mask_len))

                img[top_left_y:(top_left_y + self.mask_len),
                    top_left_x:(top_left_x + self.mask_len), :] = self.color
        else:
            # randomly select patches of the image to mask
            patch_ind = np.arange(0, h, self.patch_len)
            patch_coords = list(product(patch_ind, patch_ind))
            n_masks = int(self.pct_mask * len(patch_coords))
            masked_coords = random.sample(patch_coords, n_masks)
            for coord in masked_coords:
                img[coord[0]:(coord[0] + self.patch_len),
                    coord[1]:(coord[1] + self.patch_len), :] = self.color
            
        # convert back to PIL
        img = Image.fromarray(img)

        return img
    
class GridAugment:
    def __init__(self, prob, col=0, lwd=1, pct_grid=0.05, uniform=False):
        self.prob = prob
        self.col = col
        self.lwd = lwd
        self.pct_grid = pct_grid
        self.uniform = uniform

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        
        img = img.copy() # can probably skip this?
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # first attempt at grid augment; decided to make grid lines random
        if self.uniform:
            spacing = int(1 / self.pct_grid)
            for x in range(0, width, spacing):
                draw.line([(x, 0), (x,height)], fill=self.col, width=self.lwd)

            for y in range(0, height, spacing):
                draw.line([(0, y), (width,y)], fill=self.col, width=self.lwd)                
        
        # randomly select pixel indices on each axis for grid lines
        else:
            n_lines = int(width * self.pct_grid)
            x_lines = random.sample(range(width), n_lines)
            y_lines = random.sample(range(width), n_lines)

            for x in x_lines:
                draw.line([(x, 0), (x,height)], fill=self.col, width=self.lwd)

            for y in y_lines:
                draw.line([(0, y), (width,y)], fill=self.col, width=self.lwd)

        return img
    
class AnnotationAugment:
    def __init__(self, config):
        self.config = config

    def __call__(self, img):
        add_text = random.random() < self.config['text_prob']
        add_shape = random.random() < self.config['shape_prob']
            
        if not (add_text or add_shape):
            return img

        img = img.copy() # can probably skip this?
        draw = ImageDraw.Draw(img)
        
        width, height = img.size
        center = (width / 2, height / 2)

        # calculate for random uniform annotation position
        x_min = int(width * (1 - self.config['x_center_pct']) / 2)
        x_max = int(width * (1 + self.config['x_center_pct']) / 2)
        y_min = int(height * (1 - self.config['y_center_pct']) / 2)
        y_max = int(height * (1 + self.config['y_center_pct']) / 2)

        # add random text to image 
        font_list = ['26165_MPM_____.ttf', '16020_FUTURAM.ttf', '39335_UniversCondensed.ttf', '02587_ARIALMT.ttf', '07558_CenturyGothic.ttf']
        skewed_distribution = self.config['skewed_distribution']
        probabilities = self.config['probabilities']
        
        if add_text:
            for _ in range(random.randint(1, self.config['max_text'])):
                # randomize:
                if self.config['position'] == 'gaussian':
                    # position with bias towards middle
                    text_position = (center[0] + random.gauss(0, self.config['std_dev']), center[1] + random.gauss(0, self.config['std_dev']))
                elif self.config['position'] == 'uniform':
                    # position with uniform distribution within center bounds of image
                    text_position = (random.randint(x_min, x_max), random.randint(y_min, y_max))
                # word from random string
                length = random.choices(skewed_distribution, probabilities)[0]
                text = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                # font
                font_path = os.path.join(self.config['font_path'], random.choice(font_list))
                font  = ImageFont.truetype(font_path, 20, encoding="unic")
    
                draw.text(text_position, text, fill=color, font=font)                

        # add random shape to image
        if add_shape:
            for _ in range(random.randint(1, self.config['max_shapes'])):
                # randomize:
                if self.config['position'] == 'gaussian':
                    # position with bias towards middle. the polygon is inscribed in a bounding circle (x, y, r)
                    shape_center = (center[0] + random.gauss(0, self.std_dev), center[1] + random.gauss(0, self.std_dev))
                elif self.config['position'] == 'uniform':
                    # position with uniform distribution within center bounds of image
                    shape_center = (random.randint(x_min, x_max), random.randint(y_min, y_max))
                shape_radius = random.randint(self.config['min_shape_r'], self.config['max_shape_r'])
                bounding_circle = (shape_center, shape_radius)
                    # number of sides of the polygon. e.g. 3=triangle, 6=hexagon
                n_sides = random.randint(self.config['min_shape_sides'], self.config['max_shape_sides'])
                rotation = random.randint(0,359)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                draw.regular_polygon(bounding_circle, n_sides = n_sides, rotation=rotation, fill=color)
            
        return img