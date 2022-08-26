import torch
from torch.utils.data import DataLoader
from datasets.dataloader import UnetDataloader
from models.unet import Unet
from models.metrics import * 
from utils.plots import *


PATH_TEST = './models/unetm_TEST-ep25-4.pth'
TEST_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/test'
batch_size = 4

classes = {'GM': 1, 'WM': 2, 'CSF': 3}
batch_dice = {'GM': None, 'WM': None, 'CSF': None}
gen_dice = {'GM': 0.0, 'WM': 0.0, 'CSF': 0.0}
num_classes = len(classes)

val_heads = 2
train_heads = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

dices = []

for heads in range(train_heads):

    PATH_SUPERVISED = './models/unetm_wts-h'+str(heads+1)+'-ep25-4.pth'

    test_ds = UnetDataloader(
        TEST_PATH,
        'T1.nii',
        'LabelsForTraining.nii',
        48, 
        val_heads
    )

    test_mris = DataLoader(
        test_ds, 
        batch_size=batch_size,
    )

    unet = Unet(num_classes=4, depth=5).to(device, dtype=torch.double)
    unet.load_state_dict(torch.load(PATH_SUPERVISED))

    # since we're not training, we don't need to calculate the gradients for our outputs
    unet.eval()
    with torch.no_grad():
        for i, data in enumerate(test_mris):
            images, labels = data
            images = images.unsqueeze(1).to(device, dtype=torch.double)
            labels = labels.to(device, dtype=torch.double)

            # calculate outputs by running images through the network
            outputs = unet(images)
            pval = probs(outputs)
            preds = torch.argmax(pval, dim=1)

            for key, value in classes.items():
                batch_dice[key] = dice_coeff(torch.where(preds==value, 1, 0), 
                                  torch.where(labels==value, 1, 0)
                )
                gen_dice[key] += batch_dice[key].item()
                #print(f'Test {key} Dice score (batch): {batch_dice[key].item()}')

            #plot_batch_full(images.squeeze(1), labels, preds)
    gen_dice = {k: v / (i+1) for k, v in gen_dice.items()}
    total = sum(gen_dice.values())/num_classes
    dices.append(total)
    print(f'{gen_dice.values()}, {total:.3f}')

dices = torch.tensor(dices)
torch.save(dices, PATH_TEST)