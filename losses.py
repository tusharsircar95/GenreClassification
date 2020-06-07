import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


def triplet_loss(embeddings,y,margin=5):
  pw_dot = torch.matmul(embeddings,torch.transpose(embeddings,0,1))
  sq_norm = torch.diag(pw_dot)
  pw_dist = sq_norm.unsqueeze(1) + sq_norm.unsqueeze(0) - 2*pw_dot


  anchor_labels = y.unsqueeze(1).repeat(1,y.size(0))
  other_labels = y.unsqueeze(0).repeat(y.size(0),1)

  positive_mask = (anchor_labels == other_labels).type(torch.float)
  anchor_positive_dist = positive_mask * pw_dist
  hardest_positive_dist, _ = torch.max(anchor_positive_dist,dim=1)
  
  negative_mask = (anchor_labels != other_labels).type(torch.float)
  max_dist, _ = torch.max(pw_dist,axis=1,keepdims=True)
  anchor_negative_dist = pw_dist + (1.0-negative_mask)*max_dist
  hardest_negative_dist, _ = torch.min(anchor_negative_dist,dim=1)

  triplet_dist = hardest_positive_dist - hardest_negative_dist + margin
  triplet_loss = torch.mean(torch.nn.functional.relu(triplet_dist))
  return triplet_loss



import torch
import torch.nn as nn

class CenterLoss(nn.Module):
  """Center loss.
  
  Reference:
  Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
  
  Args:
      num_classes (int): number of classes.
      feat_dim (int): feature dimension.
  """
  def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
      super(CenterLoss, self).__init__()
      self.num_classes = num_classes
      self.feat_dim = feat_dim
      self.use_gpu = use_gpu

      if self.use_gpu:
          self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
      else:
          self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

  def forward(self, x, labels):
      """
      Args:
          x: feature matrix with shape (batch_size, feat_dim).
          labels: ground truth labels with shape (batch_size).
      """
      batch_size = x.size(0)
      distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
      distmat.addmm_(1, -2, x, self.centers.t())

      classes = torch.arange(self.num_classes).long()
      if self.use_gpu: classes = classes.cuda()
      labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
      mask = labels.eq(classes.expand(batch_size, self.num_classes))

      dist = distmat * mask.float()
      loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

      return loss


class SphereLoss(nn.Module):
    def __init__(self ):
        super(SphereLoss, self).__init__()

        # Parameters for computing loss function
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)
        
        
        # IMPLEMENT loss
        
        # Compute Lambda 
        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        
        # Generate OHE to get the label indices
        
        ohe_index = torch.zeros(cos_theta.data.shape).cuda()#cos_theta.data * 0.0 #torch.zeros(cos_theta.shape).cuda()
        ohe_index = Variable(ohe_index.scatter(dim=1,index=target.data.view(-1,1),value=1))
        
        
        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        
        output = cos_theta * 1.0 #size=(B,Classnum)
        ouput = output + ohe_index * (phi_theta*(1.0+0)/(1+self.lamb))
        output = output - ohe_index * ((cos_theta*(1.0+0))/(1+self.lamb))
        
        target = target.view(-1)
       
        loss = torch.nn.CrossEntropyLoss()(output,target)
        

        _, predictedLabel = torch.max(cos_theta.data, 1)
        predictedLabel = predictedLabel.view(-1, 1)
        accuracy = (predictedLabel.eq(target.view(-1,1).data).cpu().sum().item() ) / float(target.size(0) )

        return loss, accuracy


class CosLoss(nn.Module):
    def __init__(self, s=64 ):
        super(CosLoss, self).__init__()
        self.s = s

    def forward(self, input, target):
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        # IMPLEMENT loss
        
        # Generate OHE to get the label indices
        
        ohe_index = torch.zeros(cos_theta.data.shape).cuda()#cos_theta.data * 0.0 #torch.zeros(cos_theta.shape).cuda()
        ohe_index = Variable(ohe_index.scatter(dim=1,index=target.data.view(-1,1),value=1))
        
        output = (cos_theta * (1.0 - ohe_index)) * self.s
        output = output + (phi_theta * ohe_index) * self.s
        loss = torch.nn.CrossEntropyLoss()(output,target.view(-1))
        
        _, predictedLabel = torch.max(cos_theta.data, 1)
        predictedLabel = predictedLabel.view(-1, 1)
        accuracy = (predictedLabel.eq(target.view(-1,1).data).cpu().sum().item() ) / float(target.size(0) )

        return loss, accuracy
