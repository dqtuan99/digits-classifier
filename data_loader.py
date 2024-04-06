# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:49:04 2024

@author: Tuan
"""

import torch
from torchvision import transforms


class ImgData(torch.utils.data.Dataset):
    def __init__(self, path, w, h):

        self.transform = transforms.Compose([transforms.Resize([w, h]),
                                             transforms.Normalize([0.5], [0.5])])

        self.data = torch.load(path)

        self.img = self.transform(self.data[0])
        self.img = self.pre_processing(self.img)

        self.label = self.data[1]

        self.len = self.label.shape[0]


    def pre_processing(self, img):
        if len(img.shape) < 4:
            img = img.unsqueeze(1).repeat(1, 3, 1, 1)

        return img

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        return self.img[index], self.label[index]