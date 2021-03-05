import numpy as np
from skimage import io, transform
import pandas as pd

class P2DataLoader():

    def __init__(self, csv_file, root='', train=True, transform=None,):
        
        # path to image data
        self.csv_data = pd.read_csv(csv_file)
        self.target = np.array(self.csv_data['Sex (subj)']) # label of image
        self.im_file = np.array(self.csv_data['Image File']) # label of image
        self.h = np.array(self.csv_data['Image Height'])
        self.w = np.array(self.csv_data['Image Width'])
        self.x1 = np.array(self.csv_data['X (top left)'])
        self.x2 = np.array(self.csv_data['X (bottom right)'])
        self.y1 = np.array(self.csv_data['Y (top left)'])
        self.y2 = np.array(self.csv_data['Y (bottom right)'])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        #make slicer from bbox
        img, target, h, w = io.imread(f'{root}/{self.im_file[index]}'), self.targets[index], self.h[index], self.w[index]
        # slicer from bbox
        # might have my x's and y's backwards
        img = img[self.y1[index]:self.y2[index],self.x1[index]:self.x2[index]]
        # resize to a standard size
        img = img.resize((89, 80), Image.ANTIALIAS)
        
        """
        I have a transform library we can use here
        """
#         if self.transform is not None:
#             img = self.transform(img)

        out = {'image': img,
               'target': target,
               'meta': {'im_size': (h, w), 'index': index, 'class_ID': target}}

        return out

    def get_image(self, index):
        img = index
        return img

    def __len__(self):
        return len(self.data)
