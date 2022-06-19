import os
import pandas as pd
import numpy as np

from skimage import io
from torch.utils.data import Dataset

class SARImageDataset(Dataset):
    def __init__(self, train_dir = 'data/train4', target_dir = 'data/reference', transform=None, target_transform=None, train = True):
        self.train_dir = train_dir
        self.target_dir = target_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __len__(self):
        if self.train:
            return 5000
        else: 
            return 1250

    def __getitem__(self, idx):
        if not self.train:
            idx = idx + 5000
            
        train_path = os.path.join(self.train_dir, 'train_' + str(idx) + '.tif')
        target_path = os.path.join(self.target_dir, 'reference_' + str(idx) + '.tif')

        train_image = io.imread(train_path)
        target_image = io.imread(target_path)

        train_image = np.expand_dims(train_image, 2).astype(np.uint8) # astype needed for totensor to work
        target_image = np.expand_dims(target_image, 2).astype(np.uint8)

        if self.transform:
            train_image = self.transform(train_image)
        if self.target_transform:
            target_image = self.target_transform(target_image)
        return train_image, target_image