    

# Packages
import torch
import torchvision
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import itertools
import pandas as pd

from model.modelBase import *
from model.wrapperModel import *
from utils.evalUtils import *
from ProcessAndAugment import *

# paths
datadir = "./data"
inputdir= datadir + "/raw"
outputdir= datadir + "/processed"

# Parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size=128

n_grapheme = 168
n_vowel = 11
n_consonant = 7
n_total = n_grapheme + n_vowel + n_consonant
print('n_total', n_total)

# Model Selection

# core model
predictor = densenet(in_channels=1, out_dim=n_total).to(device)
predictor.requires_grad = True
print('predictor', type(predictor))

# select our wrapper class
classifier = BengaliClassifier(predictor).to(device)
classifier.requires_grad = True
print('classifier',type(classifier))

# Model Parameters
epochs = 3
lr = .01 # TODO: starting with flat LR, but need to implement scheduler
bs = 64

optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)


#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-10)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)

validate_every = 5 # TODO: validate every n batches or epochs
checkpoint_every = 5 # TODO: implement model checkpoints


# load train file and generate dataset
train = pd.read_csv(datadir+'/train.csv')
indices = [0] # just set to list of all indices when actually training
dataset, crop_rsz_img = genDataset(indices, inputdir, data_type = "train", train = train) # generates the dataset class

# our weights for the weighted random sampler for each epoch
consonant_weights = genWeightTensor("consonant_diacritic", train[:len(crop_rsz_img)])
root_weights = genWeightTensor("grapheme_root", train[:len(crop_rsz_img)])
vowel_weights = genWeightTensor("vowel_diacritic", train[:len(crop_rsz_img)])
grapheme_weights = genWeightTensor("grapheme", train[:len(crop_rsz_img)])

weights = {"consonant_diacritic": consonant_weights,
           "grapheme_root": root_weights,
           "vowel_diacritic": vowel_weights,
           "grapheme": grapheme_weights}

#weight_keys = list(weights.keys())
# can change the focus of the sampler like so
weight_keys = ['grapheme', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme', 'grapheme_root']

# testing without sampler for now
train_loader = DataLoader(dataset, batch_size=bs, shuffle = True)
num_batches = len(train_loader)

for i, wkey in zip(range(epochs), itertools.cycle(weight_keys)):
    print(i, wkey)
    
    # generate sampler and loader specific to epoch
    wgt_val = weights[wkey]
    #sampler = WeightedRandomSampler(wgt_val, len(wgt_val))
    #train_loader = DataLoader(dataset, batch_size=bs, sampler=sampler)
    
    # init
    predictor.train()
    classifier.train()
    
    # store accuracy results
    acc_root = []
    acc_consonant = []
    acc_vowel = []
    
    for j, (images, labels) in enumerate(train_loader):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        
        # reset
        optimizer.zero_grad()
        
        # run model - requires 4d float input
        loss, metrics, pred = classifier(images.unsqueeze(1).float(), labels)
        
        # compute loss and step
        loss.backward()
        
        optimizer.step()
        
        
        # store metrics
        acc_root.append(metrics['acc_grapheme'].to("cpu").numpy())
        acc_consonant.append(metrics['acc_consonant'].to("cpu").numpy())
        acc_vowel.append(metrics['acc_vowel'].to("cpu").numpy())
        #print(metrics)
        #break
    
    print("Epoch Metrics")
    print(f"grapheme root accuracy: {np.mean(acc_root)}")
    print(f"consonant diacritic accuracy: {np.mean(acc_consonant)}")
    print(f"vowel diacritic accuracy: {np.mean(acc_vowel)}")
    

torch.save(classifier.state_dict(),'./savedModels/densenet_tmp.pth')