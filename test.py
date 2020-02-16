# Packages
import torch
import torchvision
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable


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

epochs = 10
lr = .01 # TODO: starting with flat LR, but need to implement scheduler
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

model = densenet().to(device)
state_dict =torch.load('densenet_saved_weights.pth')
predictor=load_my_state_dict(model, state_dict)

classifier = BengaliClassifier(predictor).to(device)




#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-10)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)

validate_every = 5 # TODO: validate every n batches or epochs
checkpoint_every = 5 # TODO: implement model checkpoints


# load train file and generate dataset
test = pd.read_csv(datadir+'/test.csv')
indices = [0,1,2,3] # just set to list of all indices when actually training
submission = True
data_type = 'test'
dataset = genDataset(indices, inputdir, data_type = "test") # generates the dataset class


# testing without sampler for now
test_loader = DataLoader(dataset, batch_size=bs, shuffle = True)
num_batches = len(test_loader)

model.eval()
predictions = []
with torch.no_grad():
    for idx, (inputs) in tqdm(enumerate(test_loader),total=len(test_loader)):
        inputs.to(device)
            
        outputs1,outputs2,outputs3 = classifier(inputs.unsqueeze(1).float().cuda())
        predictions.append(outputs3.argmax(1).cpu().detach().numpy())
        predictions.append(outputs2.argmax(1).cpu().detach().numpy())
        predictions.append(outputs1.argmax(1).cpu().detach().numpy())



print(predictions)

