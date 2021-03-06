# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 22:27:29 2021

@author: Zirak
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()



set_name1 = "X_small_train.csv"
url = 'https://storage.googleapis.com/uga-dsp/project2/files/' + set_name1
landmarks_frame = pd.read_csv(url, header = None)

n = 1
img_name = landmarks_frame.iloc[n, 2]
landmarks = landmarks_frame.iloc[n, 9:145]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)




def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('https://storage.googleapis.com/uga-dsp/project2/images/', img_name)),
               landmarks)
plt.show()







class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 2])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 9:145]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
face_dataset = FaceLandmarksDataset(csv_file='https://storage.googleapis.com/uga-dsp/project2/files/X_small_train.csv',
                                    root_dir='https://storage.googleapis.com/uga-dsp/project2/images/')



fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break
    



