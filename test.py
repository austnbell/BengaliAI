# Packages
import torch
import torchvision
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd
from tqdm import tqdm

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

classifier = BengaliClassifier(predictor)
classifier.load_state_dict(state_dict)
classifier = classifier.to(device)


# load train file and generate dataset
test = pd.read_csv(datadir+'/test.csv')
indices = [0,1,2,3] # just set to list of all indices when actually training
submission = True
data_type = 'test'
dataset = genDataset(indices, inputdir, data_type = "test") # generates the dataset class

# push to data loader
test_loader = DataLoader(dataset, batch_size=bs, shuffle = False)

predictor.eval()
classifier.eval()
predictions = []
with torch.no_grad():
    for inputs in tqdm(test_loader):
        inputs = inputs.to(device)
            
        outputs1,outputs2,outputs3 = classifier(inputs.unsqueeze(1).float())
        
        predictions.append(outputs3.argmax(1).cpu().detach().numpy())
        predictions.append(outputs2.argmax(1).cpu().detach().numpy())
        predictions.append(outputs1.argmax(1).cpu().detach().numpy())



print(predictions)
