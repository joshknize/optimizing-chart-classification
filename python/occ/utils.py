import os
import random

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torch import nn


def seed_all(seed):
    '''
    Seed all to ensure complete reproducibility of results
    '''
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def print_elements(data, indent=0):
    '''
    Print the elements of the json config file into the log
    '''
    if isinstance(data, dict):
        for key, value in data.items():
            print(' ' * indent + str(key) + ": " + str(value))
            print_elements(value)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            print(' ' * indent + "[" + str(index) + "]: ")
            print_elements(item, indent + 2)
    else:
        print(' ' * indent + str(data))

# def hook_attn_map(input, output, attn_maps, n_heads):

#     with torch.no_grad():
#         input = input[0]
#         B, N, C = input.shape
#         qkv = (
#             output.detach()
#             .reshape(B, N, 3, 12, C // n_heads)
#             .permute(2, 0, 3, 1, 4)
#         )
#         q, k = (
#             qkv[0],
#             qkv[1]
#         )
#         attn = (q @ k.transpose(-2, -1)) * (C // n_heads)
#         attn_maps.append(attn)

def make_hook(attn_maps, n_heads):
    def hook(module, input, output):
        with torch.no_grad():
            input = input[0]
            B, N, C = input.shape
            qkv = (
                output.detach()
                .reshape(B, N, 3, n_heads, C // n_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k = qkv[0], qkv[1]
            attn = (q @ k.transpose(-2, -1)) * (C // n_heads)
            attn_maps.append(attn)
    return hook

def overlay_attn(attn_map, head='agg', size=518, patch_len=14):
    
    n_patches = size / patch_len
    size = (size,size)
    attn_map = attn_map[:, 0, 1:]

    if not head == 'agg':
        # option to choose the attention map from a single head of ViT
        attn_map = attn_map[head]
    else:
        # default to an average of the attention across all heads
        attn_map = attn_map.mean(axis=0)

    # normalize values
    with torch.no_grad():
        layer_norm = nn.LayerNorm(attn_map.size(), eps=1e-6)
        if torch.cuda.is_available():
            layer_norm = layer_norm.to('cuda')
        attn_map = layer_norm(attn_map)

    # need to remove register and classification indices
    num_registers = int(attn_map.shape[-1] - (n_patches * n_patches))
    attn_map = attn_map[num_registers:]

    # now get softmax
    attn_map = attn_map.softmax(dim=-1)

    attn_map = attn_map.view((int(n_patches), int(n_patches))) # reshape to patch grid

    attn_map = torch.nn.functional.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        size=size[0],  # Target size
        mode='bilinear',  # Bilinear interpolation
        align_corners=False
    ).squeeze(0).squeeze(0)

    return attn_map

# this needs to be module top-level to keep serialization through pickle possible
def get_resize_xfrm(size, resizer_helper, fill=255, method='resize_keep_aspect_ratio'):

    transforms_size = []

    if method == 'crop':
        # built in torchvision functions are picklable
        transforms_size.append(transforms.RandomCrop(size, pad_if_needed=True, fill=fill))
    elif method == 'resize':
        transforms_size.append(transforms.Resize(size))
    elif method == 'resize_keep_aspect_ratio':
        # transforms.Resize scales smallest dim to `size` arg
        transforms_size.append(transforms.Lambda(resizer_helper.resize_on_smaller_dim))
        # pad with white to ViT input size
        transforms_size.append(transforms.Lambda(resizer_helper.pad_image))

    ### needs work to be picklable
    # elif method == 'prop_crop':
    #     transforms_size = [
    #         transforms.Lambda(lambda img: transforms.functional.crop(
    #             img,
    #             top=random.randint(0, int(img.size[1] * 0.5)),
    #             left=random.randint(0, int(img.size[0] * 0.5)),
    #             height=int(img.size[1] * 0.5),
    #             width=int(img.size[0] * 0.5))),
    #         transforms.Resize(size)
    #     ]
    # elif method == 'systematic_crop':
    #     resize = size[0]
    #     transforms_size = [
    #         transforms.Lambda(lambda img: systematic_crop(img, epoch, resize))
    #     ]

    return transforms_size
