# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:37:11 2020

Class functions for our baseline model 

This model first predicts the whole grapheme
Then uses the predictions from the whole grapheme as an input to predicting the three components

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
import pretrainedmodels
from collections import OrderedDict

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
                 use_bn=True, activation=F.relu, dropout_ratio=.5, residual=False,):
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
        # compare DenseNet and SeResNet
        self.base_model = models.densenet121(pretrained=True)
        inch = self.base_model.classifier.in_features
        #self.base_model = pretrainedmodels.__dict__['se_resnext101_32x4d'](pretrained="imagenet")
        #inch = self.base_model.last_linear.in_features
        
        # should move to train parameters
        activation = F.leaky_relu
        hdim = 512
        n_total_graphemes = 1285
        
        # add add extra layer in front of first linear block
        # input and output parameter still not work!!!!
        layer0_modules = [
                ('conv1', nn.Conv2d(3, inch, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, 128, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(128)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))

        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        
        self.lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, 
                                dropout_ratio = .1, residual=False)
        
        # predicts the whole grapheme 
        # the out dimension is now the number of classes for whole grapheme prediction
        self.lin2 = LinearBlock(hdim, n_total_graphemes, use_bn=use_bn, activation=None, residual=False)
        
        # the input is the concatenation of lin1 and lin 2
        # input = h_dim + out_dim_lin2; output = out_dim
        self.lin3 = LinearBlock(hdim + n_total_graphemes, out_dim, use_bn=False, activation=None, residual=False)
        
        #self.lin_layers = Sequential(lin1, lin2, lin3)

    # the core forward pass
    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h) # I want to make sure that this is correct
        h = torch.sum(h, dim=(-1, -2)) # pooling function 
        
        h = self.layer0(h)
        
        # take out of loop and write out manually
        h1 = self.lin1(h)
        h_grapheme = self.lin2(h1)
        out = self.lin3(torch.cat((h1, h_grapheme), 1))
       
        
        return out, h_grapheme
