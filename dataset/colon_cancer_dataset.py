from collections import namedtuple
from functools import partial
import hashlib
import os
from PIL import Image
import torch
import urllib.request
from os import path
import sys
import zipfile
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import scipy.io
import pdb
import matplotlib.pyplot as plt
import imageio


class ColonCancerDataset(Dataset):

    CLASSES = [0, 1]

    def __init__(self, directory, train=True):
        cwd = os.getcwd().replace('dataset', '')
        directory = path.join(cwd, directory)

        self.data = [os.path.join(directory, x) for x in os.listdir(directory)]

        if train:
            self.image_transform = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                                                       transforms.ToTensor()
                                                       # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                       ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        folder_path = self.data[i]
        img_id = int(folder_path.split('/')[-1].replace('img', ''))

        mat = scipy.io.loadmat(path.join(folder_path, f'img{img_id}_epithelial.mat'))['detection']
        x_high = imageio.imread(path.join(folder_path, f'img{img_id}.bmp'))

        x_high = self.image_transform(x_high)
        x_low = F.interpolate(x_high[None, ...], scale_factor=0.2, mode='bilinear')[0]

        category = int(mat.shape[0] > 0)
        return x_low, x_high, category

    def strided(self, N):
        """Extract N images almost in equal proportions from each category."""
        order = np.arange(len(self.data))
        np.random.shuffle(order)
        idxs = []
        cat = 0
        while len(idxs) < N:
            for i in order:
                _, _, category = self[i]
                if cat == category:
                    idxs.append(i)
                    cat = (cat + 1) % len(self.CLASSES)
                if len(idxs) >= N:
                    break
        return idxs


if __name__ == '__main__':
    colon_cancer_dataset = ColonCancerDataset('colon_cancer', train=True)
    print()