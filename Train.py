    

# Packages
import torch
import torchvision
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping
import itertools
import pandas as pd

#from model.modelBase import *
from model.modelV2 import *
from model.wrapperModel import *
from utils.evalUtils import *
from ProcessAndAugment import *

# paths
datadir = "./data"
inputdir= datadir + "/raw"
outputdir= datadir + "/processed"

# Parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
epochs = 10
lr = .001 # TODO: starting with flat LR, but need to implement scheduler
bs = 64

optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)


#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-10)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)

validate_every = 5 # TODO: validate every n batches or epochs
checkpoint_every = 5 # TODO: implement model checkpoints


# load train file and generate dataset
train = pd.read_csv(datadir+'/train.csv')
train = convertGrapheme(train) # generate our grapheme labels
indices = [0,1,2,3] # just set to list of all indices when actually training
dataset, crop_rsz_img = genDataset(indices, inputdir, data_type = "train", train = train) # generates the dataset class

# Split data to training and validation
valid_size = 0.2
num_train = len(dataset)
train_indices = list(range(num_train))
np.random.shuffle(train_indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = train_indices[split:], train_indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)



 
# obtain training indices that will be used for validation
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
train_loader = DataLoader(dataset, batch_size=bs, sampler=train_sampler)
valid_loader = DataLoader(dataset,batch_size=bs,  sampler=valid_sampler)

# initialize the early_stopping object
patience = 20
early_stopping = EarlyStopping(patience=patience, verbose=True)

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

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    #recall
    train_recall = []
    valid_recall = []
    
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
        train_recall.append(metrics['weighted_recall'])
        train_losses.append(loss.item())
        #print(metrics)
        #break

    # validation
    predictor.eval()
    classifier.eval()

    with torch.no_grad():
        for j,(images, labels) in enumerate(valid_loader):
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            
            loss, metrics, pred = classifier(images.unsqueeze(1).float(), labels)
            valid_losses.append(loss.item())
            valid_recall.append(metrics['weighted_recall'])


    # calculate average loss over an epoch
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    print_msg = (f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f} ' +
                 f'train_recall: {np.mean(train_recall):.5f} ' +
                 f'valid_recall: {np.mean(valid_recall):.5f} ')
        
    # clear lists to track next epoch
    train_losses = []
    valid_losses = []
       
    early_stopping(valid_loss, classifier)

    if early_stopping.early_stop:
        print("Early stopping")
        break
    
    print("Epoch Metrics")
    print(print_msg)
    print(f"grapheme root accuracy: {np.mean(acc_root)}")
    print(f"consonant diacritic accuracy: {np.mean(acc_consonant)}")
    print(f"vowel diacritic accuracy: {np.mean(acc_vowel)}")
    

torch.save(classifier.state_dict(),'./savedModels/whole_grapheme.pth')
