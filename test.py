# Packages
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from model.modelBase import *
from model.wrapperModel import *
from utils.evalUtils import *

from ProcessAndAugment import *

# paths
datadir = "./data"
inputdir= datadir + "/raw"
outputdir= datadir + "/processed"
data_type = 'test'
# Parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size=128

n_grapheme = 168
n_vowel = 11
n_consonant = 7
n_total = n_grapheme + n_vowel + n_consonant

bs = 64

# Model Selection

# core model
def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

predictor = densenet(in_channels=1, out_dim=n_total).to(device)
state_dict =torch.load('./savedModels/densenet_saved_weights.pth')
#predictor=load_my_state_dict(model, state_dict)

classifier = BengaliClassifier(predictor,data_type='test')
classifier.load_state_dict(state_dict)
classifier = classifier.to(device)


# load train file and generate dataset
test = pd.read_csv(datadir+'/test.csv')
indices = [0,1,2,3] # just set to list of all indices when actually training
submission = True
dataset = genDataset(indices, inputdir, data_type = "test") # generates the dataset class

# push to data loader
test_loader = DataLoader(dataset, batch_size=bs, shuffle = False)


# run 
predictor.eval()
classifier.eval()

grapheme_list = []
vowel_list = []
consonant_list = []

with torch.no_grad():
    for inputs in tqdm(test_loader):
        inputs = inputs.to(device)
            
        grapheme,vowel,consonant = classifier(inputs.unsqueeze(1).float())
        
        grapheme_list += list(grapheme.argmax(1).cpu().detach().numpy())
        vowel_list += list(vowel.argmax(1).cpu().detach().numpy())
        consonant_list += list(consonant.argmax(1).cpu().detach().numpy())

###submission
row_id = []
target = []
for i in tqdm(range(len(grapheme_list))):
    row_id += [f'Test_{i}_grapheme_root', f'Test_{i}_vowel_diacritic',
               f'Test_{i}_consonant_diacritic']
    target += [grapheme_list[i], vowel_list[i], consonant_list[i]]
submission_df = pd.DataFrame({'row_id': row_id, 'target': target})
submission_df.to_csv('submission.csv', index=False)