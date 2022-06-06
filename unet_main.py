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

n_epochs = 25
batch_size = 4
lr = 0.0001

TRAIN_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/train'
VAL_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/val'

torch.cuda.empty_cache()

print(f'Parametros: {n_epochs=}, {batch_size=}, {lr=}\n')

'''
Crear datasets
'''
#transform = transforms.Compose([transforms.ToTensor()])

train_mris = FewShot_Dataloader(
    TRAIN_PATH, #'/media/davidjm/Disco_Compartido/david/datasets/MRBrainS13DataNii/TrainingData', \
    'T1.nii', 
    'LabelsForTraining.nii', 
    48, 
    'training'
)

val_mris = FewShot_Dataloader(
    VAL_PATH, #'/media/davidjm/Disco_Compartido/david/datasets/MRBrainS13DataNii/TrainingData', \
    'T1.nii', 
    'LabelsForTraining.nii', 
    48, 
    'validating'
)

train_mris_dl = DataLoader(
    train_mris, 
    batch_size=batch_size,
    shuffle=True,
    #num_workers=2
)

val_mris_dl = DataLoader(
    val_mris, 
    batch_size=batch_size,
    #num_workers=2
)

print(f'Tamano del dataset: {train_mris.df.shape[0]} slices \n')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#print(f'Tamano del dataset de entrenamiento: {train_mris_dl.len()}')

unet = Unet(1, depth=5).to(device, dtype=torch.double)
#print(torch.cuda.memory_summary(device=device, abbreviated=False))

#criterion = nn.BCELoss()
criterion = DiceLoss()

optimizer = Adam(unet.parameters(), lr=lr)

for epoch in tqdm(range(n_epochs)):  # loop over the dataset multiple times

    running_loss = 0.0
    running_dice = 0.0
    torch.cuda.empty_cache()
    unet.train()
    
    for i, data in enumerate(train_mris_dl, 0):

        torch.cuda.empty_cache()

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
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 4 == 3:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 4:.3f}')
            running_loss = 0.0

    unet.eval()
    with torch.no_grad():
        for i, testdata in enumerate(val_mris_dl):
            x, y = testdata
            x = x.unsqueeze(1).to(device, dtype=torch.double)
            y = y.to(device, dtype=torch.double)

            p = unet(x)
            pval = probs(p.squeeze(1))
            dice = dice_coeff(pval, y)
            running_dice += dice
            print(f'Val dice = {dice.item():.3f}')
        
        print(f'\nDice promedio epoca {epoch+1}: {running_dice/i:.3f}\n')


PATH = './models/unet_weights_.pth'
torch.save(unet.state_dict(), PATH)
print('Finished Training')



