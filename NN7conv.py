import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import numpy as np

mobilenet = torchvision.models.mobilenet_v2(True)
#print(net)
encoder = nn.Sequential(*(list(mobilenet.children())[:-1]))
#print(encoder)
#x = torch.rand(1,3,320,320)


class Upsample(nn.Module):
    def __init__(self,inp):
        super(Upsample,self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(inp, inp, 7, 1, 3),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inp, int(inp / 2), 1),
            nn.BatchNorm2d(int(inp / 2)),
            nn.ReLU6(inplace=True)
        )
    def forward(self,x):
        x = self.upsample(x)
        return x



class DepthPredictionNet(nn.Module):
    def __init__(self):
        super(DepthPredictionNet,self).__init__()
        self.net = encoder
        self.up1 = Upsample(1280)
        self.up2 = Upsample(640)
        self.up3 = Upsample(320)
        self.up4 = Upsample(160)
        self.up5 = Upsample(80)
        self.pw = nn.Sequential(nn.Conv2d(40, 1, 1),
                                nn.ReLU6(inplace=True))




    def forward(self,x):
        x = self.net(x)
        x = self.up1(x)
        x = F.interpolate(input=x,scale_factor=2,mode='bilinear',align_corners=False)
        x = self.up2(x)
        x = F.interpolate(input=x,scale_factor=2,mode='bilinear',align_corners=False)
        x = self.up3(x)
        x = F.interpolate(input=x,scale_factor=2,mode='bilinear',align_corners=False)
        x = self.up4(x)
        x = F.interpolate(input=x,scale_factor=2,mode='bilinear',align_corners=False)
        x = self.up5(x)
        x = F.interpolate(input=x,scale_factor=2,mode='bilinear',align_corners=False)
        x = self.pw(x)
        return x

fullmodel = DepthPredictionNet()

for param in fullmodel.net.parameters():
    param.requires_grad = False
for param in fullmodel.up1.parameters():
    param.requires_grad = True
for param in fullmodel.up2.parameters():
    param.requires_grad = True
for param in fullmodel.up3.parameters():
    param.requires_grad = True
for param in fullmodel.up4.parameters():
    param.requires_grad = True
for param in fullmodel.up5.parameters():
    param.requires_grad = True
for param in fullmodel.pw.parameters():
    param.requires_grad = True

#if torch.cuda.is_available():
# fullmodel = fullmodel.cuda()

#model_parameters = filter(lambda p: p.requires_grad, fullmodel.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])

#def count_paramters(model):
 #   return sum(p.numel() for p in model.parameters())


