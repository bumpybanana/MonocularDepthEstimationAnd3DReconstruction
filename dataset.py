import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms




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
        rgb = transforms.ToTensor()(Image.open(rgb_path).convert("RGB"))
        depth = transforms.ToTensor()(Image.open(depth_path).convert("L"))
        #print(depth.shape)
        #print(rgb.shape)
        concat = torch.cat([rgb, depth])
        concat = transforms.ToPILImage()(concat)
        #print(concat.size)
        concat = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.RandomAffine(4.5, scale=(1, 1.5)),
                                     transforms.ToTensor()
                                     ])(concat)
        concat = torch.split(concat,3,0)
        #print(concat[0].shape, concat[1].shape)


        if self.transform is not None:
            rgb = self.transform(concat[0])
            depth = self.transform(concat[1])
            #rgb = self.transform(transforms.ToPILImage()(concat[0]))
            #depth = self.transform(transforms.ToPILImage()(concat[1]))

        return rgb, depth



