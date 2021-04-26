import os
import torch
import torchvision.datasets
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np

transformation = transforms.Compose([transforms.RandomCrop(320),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomVerticalFlip(p=0.5),
                                         transforms.RandomAffine(4.5,scale=(1, 1.5)),
                                       # transforms.ColorJitter()
                                         transforms.ToTensor()])

class DepthDataset(Dataset):
    def __init__(self,rgb_dir,depth_dir,transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.images = os.listdir(rgb_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_path = os.path.join(self.rgb_dir, self.images[index])
        depth_path = os.path.join(self.depth_dir, self.images[index])
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = np.array(Image.open(depth_path).convert("RGB"))



        if self.transform is not None:
             rgb, depth = self.transform(rgb, depth)

        return rgb, depth

#depth_dataset = DepthDataset(rgb_dir="E:/nyuv2/train_rgb",depth_dir="E:/nyuv2/train_depth")

train_data = torchvision.datasets.ImageFolder(root="E:/nyuv2/train_rgb/",transform=transformation)


