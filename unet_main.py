import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.unet import Unet
from models.metrics import * 
from datasets.dataloader import FewShot_Dataloader
from tqdm import tqdm
from utils.plots import *

n_epochs = 100
batch_size = 4
lr = 0.0001

TRAIN_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/train'
VAL_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/val'

PATH = './models/unet_weights_.pth'

torch.cuda.empty_cache()

print(f'Parametros: {n_epochs=}, {batch_size=}, {lr=}\n')

'''
Crear datasets
'''
train_mris = FewShot_Dataloader(
    TRAIN_PATH,
    'T1.nii', 
    'LabelsForTraining.nii', 
    48, 
    'training'
)

val_mris = FewShot_Dataloader(
    VAL_PATH,
    'T1.nii', 
    'LabelsForTraining.nii', 
    48, 
    'validating'
)

train_mris_dl = DataLoader(
    train_mris, 
    batch_size=batch_size,
    shuffle=True,
)

val_mris_dl = DataLoader(
    val_mris, 
    batch_size=batch_size,
)

print(f'Tamano del dataset: {train_mris.df.shape[0]} slices \n')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

unet = Unet(1, depth=5).to(device, dtype=torch.double)
#print(torch.cuda.memory_summary(device=device, abbreviated=False))

#criterion = nn.BCELoss()
criterion = DiceLoss()

optimizer = Adam(unet.parameters(), lr=lr)

best_score = 1.0

for epoch in tqdm(range(n_epochs)):  # loop over the dataset multiple times

    running_loss = 0.0
    running_dice = 0.0
    epoch_loss = 0.0
    
    print(f'\n\nEpoch {epoch + 1}\n')

    unet.train()
    
    for i, data in enumerate(train_mris_dl, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)
        labels = labels.to(device, dtype=torch.double)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = unet(inputs)
        p1 = probs(outputs.squeeze(1))
        loss = criterion(p1, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss/(i + 1)

    if epoch_loss < best_score:
        best_score = epoch_loss
        print(f'Updated weights file!')
        torch.save(unet.state_dict(), PATH)

    print(f'loss = {epoch_loss:.3f}, {best_score=:.3f}')

    unet.eval()
    with torch.no_grad():
        for j, testdata in enumerate(val_mris_dl):
            x, y = testdata
            x = x.unsqueeze(1).to(device, dtype=torch.double)
            y = y.to(device, dtype=torch.double)

            p = unet(x)
            pval = probs(p.squeeze(1))
            dice = dice_coeff(pval, y)
            running_dice += dice
            #print(f'Val dice = {dice.item():.3f}')
        
    print(f'Val dice = {running_dice/(j + 1):.3f}\n')

print('Finished Training')



