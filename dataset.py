import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import Dataset

cv2.setNumThreads(4)

class myDataset(Dataset):
    def __init__(self, dataset_name, batch_size=8):
        print ('dataset init')
        self.batch_size = batch_size
        self.data_root = '/mnt/StorageMedia/dataset/'
        self.clean_root = os.path.join(self.data_root, 'clean')
        self.done_root = os.path.join(self.data_root, 'done')
        self.dataset_name = dataset_name
        self.clean_files = sorted(os.listdir(self.clean_root))
        self.done_files = sorted(os.listdir(self.done_root))
        self.h = 256
        self.w = 256
        self.load_data()

    def __len__(self):
        return len(self.clean_files)
    
    def crop(self, img0, img1, h, w):
        np.random.seed(None)
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        return img0, img1
    
    def load_data(self):
        print ('load data')
        self.meta_data = self.clean_files

    def getimg(self, index):
        img0 = cv2.imread(os.path.join(self.clean_root, self.clean_files[index]), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img1 = cv2.imread(os.path.join(self.done_root, self.done_files[index]), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        return img0, img1
    
    def __getitem__(self, index):        
        img0, img1 = self.getimg(index)
        img0, img1 = self.crop(img0, img1, self.h, self.w)
        
        '''
        p = random. uniform(0, 1)
        if p < 0.25:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        elif p < 0.5:
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
        elif p < 0.75:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        '''

        return img0, img1