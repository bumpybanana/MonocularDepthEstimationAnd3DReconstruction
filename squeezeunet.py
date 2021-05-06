import torch
import torch.nn as nn
import numpy as np


#x = torch.rand(1, 3, 320, 320)

class DoubleConv(nn.Module):
    def __init__(self,inp, oup):
        super(DoubleConv,self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 3, 1, 1)
        self.conv2 = nn.Conv2d(oup, oup, 3, 1, 1)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU()


    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.bn(x1)
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x2 = self.bn(x2)
        x = torch.cat([x1,x2], 1)
        return x



class Downsample(nn.Module): #two fire modules

    def __init__(
            self,
            inplanes: int,
            s_plane: int,
            e_plane: int
    ) -> None:
        super(Downsample, self).__init__()
        self.inplanes = inplanes
        self.relu = nn.ReLU()
        self.squeeze1 = nn.Conv2d(inplanes, s_plane, kernel_size=1,stride =2)
        self.squeeze2 = nn.Conv2d(2*e_plane, s_plane, 1, 1)
        self.expand1x1 = nn.Conv2d(s_plane, e_plane,
                                   kernel_size=1)
        self.expand3x3 = nn.Conv2d(s_plane, e_plane,
                                   kernel_size=3, padding=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.expand1x1_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze1(x))
        #print(x.shape)
        self.concat1 = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
        #print(self.concat1.shape)

        x = self.squeeze_activation(self.squeeze2(self.concat1))
       #print(x.shape)
        self.concat2 = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ],1)
        #print(self.concat2.shape)

        x = torch.cat([
            self.concat1, self.concat2
        ],1)
        return x

class DownsampleMerge(nn.Module):

    def __init__(
            self,
            inplanes: int,
            s_plane: int,
            e_plane: int
    ) -> None:
        super(DownsampleMerge, self).__init__()
        self.inplanes = inplanes
        self.relu = nn.ReLU()
        self.squeeze1 = nn.Conv2d(inplanes, s_plane, kernel_size=1,stride =2)
        self.squeeze2 = nn.Conv2d(2*e_plane, s_plane, 1, 1)
        self.expand1x1 = nn.Conv2d(s_plane, e_plane,
                                   kernel_size=1)
        self.expand3x3 = nn.Conv2d(s_plane, e_plane,
                                   kernel_size=3, padding=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.expand1x1_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze1(x))
        #print(x.shape)
        self.concat1 = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
        #print(self.concat1.shape)

        x = self.squeeze_activation(self.squeeze2(self.concat1))
       #print(x.shape)
        x = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ],1)
        #print(self.concat2.shape)

        return x
#down = Downsample(128,32,64)
#print(down(x).shape)


class Transfire(nn.Module):
    def __init__(self,inplanes,s_plane,e_plane):
        super(Transfire,self).__init__()
        self.inplanes = inplanes

        self.squeeze1 = nn.ConvTranspose2d(inplanes, s_plane, kernel_size=1, stride=2,output_padding=1)
        self.squeeze2 = nn.ConvTranspose2d(2*e_plane, s_plane, kernel_size=1)
        self.expand1x1 = nn.ConvTranspose2d(s_plane, e_plane, kernel_size=1)
        self.expand2x2 = nn.ConvTranspose2d(s_plane, e_plane,kernel_size=2,padding=1, dilation=2) #damit 40x40 statt 39,39

        self.squeeze_activation = nn.ReLU(True)
        self.expand1x1_activation = nn.ReLU(True)
        self.expand2x2_activation = nn.ReLU(True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze1(x)) #x = ds4
        #print(x.shape)
        x = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand2x2_activation(self.expand2x2(x))
        ], 1)

        return x

class Upsample(nn.Module): #two fire modules for upsampling

    def __init__(
            self,
            inplanes: int,
            s_plane: int,
            e_plane: int
    ) -> None:
        super(Upsample, self).__init__()
        self.inplanes = inplanes
        self.relu = nn.ReLU()
        self.squeeze1 = nn.Conv2d(inplanes, s_plane, kernel_size=1)
        self.squeeze2 = nn.Conv2d(2*e_plane, s_plane, 1, 1)
        self.expand1x1 = nn.Conv2d(s_plane, e_plane,
                                   kernel_size=1)
        self.expand3x3 = nn.Conv2d(s_plane, e_plane,
                                   kernel_size=3, padding=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.expand1x1_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze1(x))
        #print(x.shape)
        self.concat1 = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
        #print(self.concat1.shape)

        x = self.squeeze_activation(self.squeeze2(self.concat1))
        #print(x.shape)
        x = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ],1)
        ##print(x.shape)

        return x






class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.ds0 = nn.Sequential(nn.Conv2d(3,64,3,1,1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(64),
                                 nn.Conv2d(64,64,3,1,1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(64)
                                 )

        self.ds1 = nn.Sequential(DoubleConv(3,64),
                                 DownsampleMerge(128, 32, 64),
                                 )

        self.ds2 = nn.Sequential(DoubleConv(3,64),
                                 DownsampleMerge(128, 32, 64),
                                 DownsampleMerge(128, 48, 128),
                                 )

        self.ds3 = nn.Sequential(DoubleConv(3,64),
                                 DownsampleMerge(128, 32, 64),
                                 DownsampleMerge(128, 48, 128),
                                 DownsampleMerge(256, 64, 256)
                                 )
        self.us1 = nn.Sequential(Upsample(1024,64,256), #512,40,40
                                Transfire(512,64,128)
                                 )

        self.us2 = nn.Sequential(Upsample(512,48,128), #1,256,80,80
                                 Transfire(256,48,64)
                                )

        self.us3 = Upsample(256,32,64)
        self.downsample = nn.Sequential(DoubleConv(3,64), #(1,128,320,320)
                                        Downsample(128,32,64), #([1, 256, 160, 160])
                                        nn.ReLU(),
                                        Downsample(256,48,128),#(1,512,80,80)
                                        Downsample(512,64,256),#(1,1024,40,40)
                                        Downsample(1024,80,512),#(1,2048, 20, 20)
                                        Transfire(2048,80,256)
                                      )
        self.convtranspose = nn.Sequential(nn.ConvTranspose2d(128,64,2,2),
                                           nn.ReLU()
                                           )
        self.lastconvs = nn.Sequential(nn.Conv2d(128,64,3,padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(64),
                                       nn.Conv2d(64,64,3,padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(64),
                                       nn.Conv2d(64,1,1),
                                       nn.Sigmoid()
                                      )


    def forward(self,x):
        ds0 = self.ds0(x)
        ds1 = self.ds1(x)
        ds2 = self.ds2(x)
        ds3 = self.ds3(x)
        x = self.downsample(x)
        x = torch.cat([x,ds3],1) #1,1024,40,40
        x = self.us1(x) #1,256,80,80
        x = torch.cat([x,ds2],1) #1,512,80,80
        x = self.us2(x) #1,128,160,160
        x = torch.cat([x,ds1],1) # 1,256,160,160
        x = self.us3(x) # 1,128,160,160
        x = self.convtranspose(x) # 1,64,320,320
        x = torch.cat([x,ds0],1) #1,128,320,320
        x = self.lastconvs(x) #1,1,320,320
        return x


#model =Model()
#print(model(x).shape)
#
#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])
#print(params)
