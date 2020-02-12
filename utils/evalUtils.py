# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:59:07 2020

eval utils

@author: Austin Bell
"""
import torch

def accuracy(y, t):
    pred_label = torch.argmax(y, dim=1)
    count = pred_label.shape[0]
    correct = (pred_label == t).sum().type(torch.float32)
    acc = correct / count
    return acc