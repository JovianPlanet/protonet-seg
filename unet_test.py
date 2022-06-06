import torch
from torch.utils.data import DataLoader
from datasets.dataloader import FewShot_Dataloader
from models.unet import Unet
from models.metrics import * 
from utils.plots import *


PATH_SUPERVISED = './models/unet_weights_.pth'
TEST_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/test'
batch_size = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

test_ds = FewShot_Dataloader(
    TEST_PATH,
    'T1.nii',
    'LabelsForTraining.nii',
    48, 
    'testing'
)

test_mris = DataLoader(
    test_ds, 
    batch_size=batch_size,
)

unet = Unet(1, depth=5).to(device, dtype=torch.double)
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
        pval = probs(outputs.squeeze(1))
        dice = dice_coeff(pval, labels)
        print(f'Test Dice score: {dice}')
        #if i==6:
        plot_batch_full(images.squeeze(1), labels, pval>0.8)
        