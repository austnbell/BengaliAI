# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:01:40 2020
Wrapper model class
takes whatever model file and runs it 
@author: Austin Bell
"""
import torch
from torch import nn
import torch.nn.functional as F

import tqdm

from utils.evalUtils import accuracy, macro_recall
from Loss import *

# wrapper class to run the entire model 
# just requires setting our predictor to our core model class 
# predictor must return predictions for each of our labels 
class BengaliClassifier(nn.Module):
    def __init__(self, predictor, n_grapheme=168, n_vowel=11, n_consonant=7,data_type='train'):
        super(BengaliClassifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant
        self.predictor = predictor
        self.data_type = data_type

        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
            'acc_grapheme', 'acc_vowel', 'acc_consonant', 'weighted_recall']

    def forward(self, x, y=None):
        pred = self.predictor(x)
        
        if isinstance(pred, tuple):
            assert len(pred) == 3
            preds = pred
        else:
            assert pred.shape[1] == self.n_total_class
            preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
           
        # compute our individual losses and generate single loss value
        # TODO: test other loss functions
        if self.data_type == 'train':
            # change cross entropy to focal loss
            loss_grapheme = FocalLoss(preds[0], y[:, 0])
            loss_vowel = FocalLoss(preds[1], y[:, 1])
            loss_consonant = FocalLoss(preds[2], y[:, 2])
            loss = loss_grapheme + loss_vowel + loss_consonant
        
        # metric summary
            metrics = {
                'loss': loss.item(),
                'loss_grapheme': loss_grapheme.item(),
                'loss_vowel': loss_vowel.item(),
                'loss_consonant': loss_consonant.item(),
                'acc_grapheme': accuracy(preds[0], y[:, 0]),
                'acc_vowel': accuracy(preds[1], y[:, 1]),
                'acc_consonant': accuracy(preds[2], y[:, 2]),
                'weighted_recall': macro_recall(pred, y) # will figure this out later
            }
        
            return loss, metrics, pred
        else:
            return preds

    # run our prediction
    def calc(self, data_loader):
        device: torch.device = next(self.parameters()).device
        self.eval()
        output_list = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                
                batch = batch.to(device)
                pred = self.predictor(batch)
                output_list.append(pred)
        output = torch.cat(output_list, dim=0)
        preds = torch.split(output, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
        return preds

    # return probabilities
    def predict_proba(self, data_loader):
        preds = self.calc(data_loader)
        return [F.softmax(p, dim=1) for p in preds]

    # return actual predictions
    def predict(self, data_loader):
        preds = self.calc(data_loader)
        pred_labels = [torch.argmax(p, dim=1) for p in preds]
        return pred_labels
