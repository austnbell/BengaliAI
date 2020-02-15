# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:37:11 2020

Class functions for our baseline model 


@author: Austin Bell
"""

# functions
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Sequential
from torchvision import models
import torchvision

###############################################################################
# Classes
##############################################################################
# this is just a skip connection (very important if we have a ton of layers)
# if we want to concatenate input with the output of the linear layer + activation
def residual_add(lhs, rhs):
    lhs_ch, rhs_ch = lhs.shape[1], rhs.shape[1]
    if lhs_ch < rhs_ch:
        out = lhs + rhs[:, :lhs_ch]
    elif lhs_ch > rhs_ch:
        out = torch.cat([lhs[:, :rhs_ch] + rhs, lhs[:, rhs_ch:]], dim=1)
    else:
        out = lhs + rhs
    return out


# block of linear functions - this is a single layer and can be changed 
# change this if we want to change our functions
class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False,):
        super(LinearBlock, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h


# core underlying model 
class densenet(nn.Module):
    def __init__(self,in_channels = 1,out_dim=10, use_bn=True):
        super(densenet, self).__init__()
        
        # convolution -- I do not know the point is for this
        # I will ignore this for now
        self.conv0 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True))
        
        # pretrained model 
        self.base_model = models.densenet121(pretrained=True)
       
        inch = self.base_model.classifier.in_features
        
        # should move to train parameters
        activation = F.leaky_relu
        hdim = 512
        
        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin1, lin2)

    # the core forward pass
    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h) # I want to make sure that this is correct
        h = torch.sum(h, dim=(-1, -2)) # pooling function 
        
        for layer in self.lin_layers:
            h = layer(h)
        return h
    
