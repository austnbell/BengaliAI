# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:06:52 2020

Functions for processing and Augmenting our data

@author: Austin Bell
"""
###################################################################
# Packages and paths
###################################################################

import gc
import os
import random
import sys
import albumentations as A
import pyarrow
import cv2
import itertools

from tqdm import tqdm
import numpy as np 
import pandas as pd 
import torch

from BengaliDataset import * # import our dataset class


# Parameters
HEIGHT = 137
WIDTH = 236
SIZE = 128

###################################################################
# Load Images
###################################################################

# during submission, we load via parquet
def prepare_image(datadir, data_type, submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(datadir + f'/{data_type}_image_data_{i}.parquet')
                         for i in indices]
    else:
        image_df_list = [pd.read_feather(datadir + f'/{data_type}_image_data_{i}.feather')
                         for i in indices]

    print('image_df_list', len(image_df_list))
    HEIGHT = 137
    WIDTH = 236
    
    #somehow the original input is inverted
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images

###################################################################
# Crop and Resize our images
###################################################################
    
# bounding box
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    
    #remove low intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

# run for all images 
def runCropRsz(images):
    crop_rsz_img = []
    for idx in range(len(images)):
        img0 = (255 - images[idx]).astype(np.float32) / 255
        #normalize each image by its max val
        img = (img0*(255.0/img0.max())).astype(np.float32)
        img = crop_resize(img)

        # add to our stored list
        crop_rsz_img.append(img)

    crop_rsz_img = np.array(crop_rsz_img)
    
    return crop_rsz_img



###################################################################
# Data Augmentations Pipeline
###################################################################
# define our augmentations
def augPipeline(P = .5):
    return A.Compose([
        A.IAAAdditiveGaussianNoise(p=.3),
        A.OneOf([
            A.MedianBlur(blur_limit=3, p=0.4),
            A.Blur(blur_limit=1, p=0.4),
        ], p=0.5),
        A.ShiftScaleRotate(shift_limit=.1, scale_limit=0.0, rotate_limit=15, p=.75),
        A.OneOf([
            A.OpticalDistortion(p=.4),
            A.GridDistortion(p=.2),
            A.IAAPiecewiseAffine(p=.5),
        ], p=.33)], p=P)
    
    
# generates weights by class to pass into a sampler during training
def genWeightTensor(column, train):
    class_counts = train[column].value_counts()
    weight = 1 / class_counts
    return torch.tensor([weight[t] for t in train[column]])



###################################################################
# Main Function to Generate and Load the dataset
###################################################################
def genDataset(indices, inputdir, data_type = "train", train = None):
    assert data_type in ['train', 'test']    
    
    submission = False if data_type == "train" else True
    indices = indices # which train files to load 
    images = prepare_image(inputdir, data_type=data_type, submission=submission, indices=indices)
    #images = images[:int(round(len(images)*.5,0))]
    print("~~Loaded Images~~")
    
    # run our crop and resize functions
    crop_rsz_img = runCropRsz(images)
    print("~~Standardized Images~~")

    # init augmentation pipeline
    augmentation = augPipeline()

    # generate our dataset
    if data_type == "train":
        train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
        dataset = BengaliAIDataset(crop_rsz_img, labels = train_labels[:len(crop_rsz_img)], transform = augmentation) 
        
        return dataset, crop_rsz_img
    else:
        dataset = BengaliAIDataset(crop_rsz_img)
        return dataset


######################################################
# Testing
#####################################################
"""
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader

# Paths
datadir = "./data"
inputdir= datadir + "/raw"
outputdir= datadir + "/processed"

# load train file and generate dataset
train = pd.read_csv(datadir+'/train.csv')
dataset, crop_rsz_img = genDataset([0], inputdir, train = train)

print(dataset.get_example(0))

# test dataloader and balancing
consonant_weights = genWeightTensor("consonant_diacritic", train[:len(crop_rsz_img)])
root_weights = genWeightTensor("grapheme_root", train[:len(crop_rsz_img)])
vowel_weights = genWeightTensor("vowel_diacritic", train[:len(crop_rsz_img)])
grapheme_weights = genWeightTensor("grapheme", train[:len(crop_rsz_img)])

weights = {"consonant_diacritic": consonant_weights,
           "grapheme_root": root_weights,
           "vowel_diacritic": vowel_weights,
           "grapheme": grapheme_weights}

keys = list(weights.keys())

for i, key in zip(range(8), itertools.cycle(keys)):
    print(i, key)
    
    # generate sampler and loader specific to epoch
    w = weights[key]
    sampler = WeightedRandomSampler(w, len(w))
    train_loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    
    for j, (x, y) in enumerate(train_loader):
        print("x.shape {}, y.shape {}".format(x.shape, y.shape))
        col = train.columns.get_loc(key) -1 if key != "grapheme" else 2
        class_count =  torch.tensor([(y[:,col] == t).sum() for t in torch.unique(y[:,col], sorted=True)])
        print(class_count)
        if j == 2:
            break

"""