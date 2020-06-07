import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np



class CosLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 0.35):
        super(CosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features) )
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / torch.clamp(xlen.view(-1,1) * wlen.view(1,-1), min=1e-8 )
        cos_theta = cos_theta.clamp(-1,1)

        # IMPLEMENT phi_theta
        phi_theta = cos_theta - self.m
        
        output = (cos_theta,phi_theta)
        return output




class SphereLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4 ):
        super(SphereLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / torch.clamp(xlen.view(-1,1) * wlen.view(1,-1), min=1e-8)
        cos_theta = cos_theta.clamp(-1,1)

        # IMPLEMENT phi_theta
        
        theta = Variable(torch.acos(cos_theta.data))
        k = torch.floor(theta/(3.14/self.m))
        cos_mtheta = self.mlambda[self.m](cos_theta)
        
        phi_theta = ((-1)**k)*cos_mtheta - 2*k
        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)

        output = (cos_theta,phi_theta)
        return output




class GenreNet(torch.nn.Module):
  def __init__(self,embed_type):
    super(GenreNet, self).__init__()
    
    self.net = models.resnet50(pretrained=True)
    
    # turn off gradients
    # for param in self.net.parameters():
    #     param.requires_grad = False

    # self.new_fc = nn.Sequential(*list(self.net.fc.children())[:-1] + [nn.Linear(2048, 10)])
    # self.net.fc = self.new_fc
    self.new_fc = nn.Sequential(*list(self.net.fc.children())[:-1])
    self.net.fc = self.new_fc

    self.embedding_layer = nn.Sequential(*list([torch.nn.Linear(2048,128),
    torch.nn.Linear(128,32)]))
    
    if embed_type == 'sphere':
      self.classifier = SphereLinear(in_features = 32,
                  out_features = 10)
    elif embed_type == 'cos':
      self.classifier = CosLinear(in_features = 32,
                out_features = 10)
    else:
      self.classifier = torch.nn.Linear(32,10)


  def forward(self,x):
    embedding = self.embedding_layer(self.net(x))
    return embedding, self.classifier(embedding)

