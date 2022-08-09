import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.unet import Unet
from models.metrics import * 
from datasets.dataloader import UnetDataloader
from tqdm import tqdm
from utils.plots import *

n_epochs = 100#0
batch_size = 4
lr = 0.0001

TRAIN_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/train'
VAL_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/val'

PATH = './models/unetmulti_weights_.pth'

classes = {'GM': 1, 'WM': 2, 'CSF': 3}
batch_dice = {'GM': None, 'WM': None, 'CSF': None}
gen_dice = {'GM': 0.0, 'WM': 0.0, 'CSF': 0.0}

num_classes = len(classes)
train_heads = 8
val_heads = 2

for heads in range(train_heads, 7, -1):

    PATH = './models/unetm_wts-h'+str(heads)+'-ep'+str(n_epochs)+'-'+str(batch_size)+'.pth'

    torch.cuda.empty_cache()

    print(f'Parametros: {n_epochs=}, {batch_size=}, {lr=}, file name={PATH[9:]}\n')

    '''
    Crear datasets
    '''
    train_mris = UnetDataloader(
        TRAIN_PATH,
        'T1.nii', 
        'LabelsForTraining.nii', 
        48, 
        heads
    )

    val_mris = UnetDataloader(
        VAL_PATH,
        'T1.nii', 
        'LabelsForTraining.nii', 
        48, 
        val_heads
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

    unet = Unet(4, depth=5).to(device, dtype=torch.double)
    #print(torch.cuda.memory_summary(device=device, abbreviated=False))

    criterion = nn.CrossEntropyLoss()
    #criterion = DiceLoss()

    optimizer = Adam(unet.parameters(), lr=lr)

    best_loss = 1.0

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
            #plot_batch(masks_pred, labels)
            
            loss = criterion(outputs, labels.long()) # Cross entropy loss performs softmax by default
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        epoch_loss = running_loss/(i + 1)        

        unet.eval()
        with torch.no_grad():
            for j, testdata in enumerate(val_mris_dl):
                x, y = testdata
                x = x.unsqueeze(1).to(device, dtype=torch.double)
                y = y.to(device, dtype=torch.double)

                outs = unet(x)
                pval = probs(outs) # Softmax
                preds = torch.argmax(pval, dim=1)

                for key, value in classes.items():
                    batch_dice[key] = dice_coeff(torch.where(preds==value, 1, 0), 
                                      torch.where(y==value, 1, 0)
                    )
                    gen_dice[key] += batch_dice[key].item()

        gen_dice = {k: v / (j+1) for k, v in gen_dice.items()}
        epoch_dice = sum(gen_dice.values())/num_classes

        if epoch == 0:
            best_loss = epoch_loss
            best_dice = epoch_dice

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        print(f'Loss = {epoch_loss:.3f}, Best loss = {best_loss:.3f}\n')

        for key, value in gen_dice.items():
            print(f'Validation {key} dice = {value:.3f}')

        if epoch_dice > best_dice:
            best_dice = epoch_dice
            print(f'\nUpdated weights file!')
            torch.save(unet.state_dict(), PATH)

        print(f'\nDice = {epoch_dice:.3f}, Best dice = {best_dice:.3f}\n')

    print('Finished Training')



