import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#x = torch.rand(1,3,320,320)
class Transfire(nn.Module):
    def __init__(self,inplanes,s_plane,e_plane):
        super(Transfire,self).__init__()
        self.inplanes = inplanes

        self.squeeze1= nn.ConvTranspose2d(inplanes, s_plane, kernel_size=1, stride=2)
        self.expand1x1 = nn.ConvTranspose2d(s_plane, e_plane, kernel_size=1)
        self.expand2x2 = nn.ConvTranspose2d(s_plane, e_plane,kernel_size=2,padding=1)

        self.squeeze_activation = nn.PReLU()
        self.expand1x1_activation = nn.PReLU()
        self.expand2x2_activation = nn.PReLU()

    def forward(self, x, concat):

        x = self.squeeze_activation(self.squeeze1(x)) 
        #print(x.shape)
        x1 = self.expand1x1_activation(self.expand1x1(x))
        x2 = self.expand2x2_activation(self.expand2x2(x))
        x1 = F.interpolate(x1, size=concat.size(2),mode='bilinear',align_corners=True)
        x2 = F.interpolate(x2, size=concat.size(2), mode='bilinear', align_corners=True)
        #print(x.shape)
        x = torch.cat([
            x1, x2
        ], 1)
        x = torch.cat([
            x, concat
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
        self.squeeze_activation = nn.PReLU()
        self.expand3x3_activation = nn.PReLU()
        self.expand1x1_activation = nn.PReLU()
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

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        self.original_model = models.mobilenet_v2(pretrained=True)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append(v(features[-1]))
        return features


class FireDecoder(nn.Sequential):
    def __init__(self):
        super(FireDecoder, self).__init__()
        self.activation = nn.PReLU()
        self.conv1 = nn.Conv2d(1280, 640, kernel_size=1, stride=1, padding=1)
        self.transfire1 = Transfire(640,80,160)
        self.upsample1 = Upsample(384,64,160)
        self.transfire2 = Transfire(320,64,80)
        self.upsample2 = Upsample(192,48,80)
        self.transfire3 = Transfire(160,48,40)
        self.upsample3 = Upsample(104,32,40)
        self.transfire4 = Transfire(80,32,20)
        self.upsample4 = Upsample(56,16,20)
        self.upconv = nn.ConvTranspose2d(40,20,kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(20,1, kernel_size=1, stride=1)


    def forward(self,features):
        encode1, encode2, encode3, encode4, encode5 = features[2], features[4], features[6], features[9], features[19]

        x = self.conv1(encode5)
        x = self.transfire1(x,encode4)
        x = self.upsample1(x)
        x = self.transfire2(x,encode3)
        x = self.upsample2(x)
        x = self.transfire3(x,encode2)
        x = self.upsample3(x)
        x = self.transfire4(x,encode1)
        x = self.upsample4(x)
        x = self.activation(self.upconv(x))
        x = self.conv2(x)
        x = self.activation(self.conv3(x))

        return x





class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = FireDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#x = torch.rand(1,3, 320, 320)
#net = Model()
#print(net(x).shape)
##print(net(x).shape)
#model_parameters = filter(lambda p: p.requires_grad, net.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])
#print(params)
