import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


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
        rgb = transforms.ToTensor()(np.array(Image.open(rgb_path).convert("RGB")))
        depth = transforms.ToTensor()(np.array(Image.open(depth_path))) / (2**16)

        concat = torch.cat([rgb, depth])
        concat = transforms.ToPILImage()(concat)



        if self.transform is not None:
            concat = self.transform(concat)
            concat = torch.split(concat,3,0)
            rgb = concat[0]
            depth = concat[1]

        return rgb, depth
