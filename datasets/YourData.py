'''
Here is an example of constructing your own dataset
'''
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
import albumentations as A
from torchvision import transforms as T
from PIL import Image
import cv2
import sys
from pathlib import PurePath

class DataBase(Dataset):
    def __init__(self, split, **kwargs):
        super().__init__()

        assert split in ['train', 'vali', 'test', 'trainval', 'traintest', 'all']

        self.split = split
    
    def _get_attr(self, index, attr):
        return self.csv.loc[index, attr]

    def __getitem__(self, index):
        # read image
        image_dir = # your image path

        image = Image.open(image_dir)
        image = np.asarray(image)

        # read mask
        mask_dir = # your mask path

        mask = Image.open(mask_dir)
        mask = np.asarray(mask)

        return image, mask

    def __len__(self):
        return # your dataset length


class CustomSeg(DataBase):
    def __init__(self, transforms, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)

        # augmentations
        data = self.transforms(image=image, mask=mask)
        image, mask = data['image'], data['mask']
        
        # wrapup
        data = {}

        data['image'] = image
        data['mask'] = mask.long()

        return data

if __name__ == "__main__":

    tfs = A.Compose(
    [
        # A.Resize(224, 288),
        A.Resize(224, 288),
        # A.Resize(256, 352),
        # A.RandomRotate90(),
        # A.RandomScale(),
        # A.RandomCrop(224, 288),
        # A.Resize(224, 224),
        # A.RandomCrop(224, 224),
        # A.CLAHE(),
        # A.RandomBrightnessContrast(p=0.5),
        # A.HorizontalFlip(p=0.5),
        # A.ElasticTransform(alpha_affine=10, p=0.1)
    ]
    )

    args = {     
        'transforms':tfs,
        'split':'train'
            }
    traindata = CustomSeg(**args)