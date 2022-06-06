import torch
import torch.nn as nn

from torch.optim import Adam, SGD

from torch.utils.data import DataLoader

from models.encoder import FS_Encoder, NeuralNetwork, FS_Encoder2
from models.unet import UnetEncoder
from models.train import train
from models.metrics import *

from datasets.dataloader import FewShot_Dataloader
from datasets.fewshot_sampler import NShotTaskSampler


evaluation_episodes = 1000
episodes_per_epoch = 15
n_epochs = 50
lr = 0.0001

TRAIN_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/train'
VAL_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/val'

'''
k clases, n muestras, q queries
'''
n_train = 8 # n shots (train)
n_test = 5 # n shots (test)
k_train = 1 # k way (k classes) (train)
k_test = 1 # k way (k classes) (test)
q_train = 5 # q queries (train)
q_test = 1 # q queries (test)

print(f'Parametros: {episodes_per_epoch=}, {n_epochs=}, {n_train=}, {k_train=}, {q_train=}, {lr=}')

'''
Crear datasets
'''

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
    batch_sampler=NShotTaskSampler(train_mris, episodes_per_epoch, n_train, k_train, q_train),
)

val_mris_dl = DataLoader(
    val_mris, 
    batch_sampler=NShotTaskSampler(val_mris, episodes_per_epoch, n_test, k_test, q_test),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

unet = UnetEncoder(1, depth=5).to(device, dtype=torch.double)

# criterion1 = nn.CrossEntropyLoss()
# criterion2 = nn.CrossEntropyLoss()

# criterion1 = nn.BCELoss()
# criterion2 = nn.BCELoss()

criterion1 = DiceLoss()
criterion2 = DiceLoss()
#optimizer = SGD(unet.parameters(), lr=0.001, momentum=0.9)
optimizer = Adam(unet.parameters(), lr=lr)

n_supp_train = n_train*k_train
n_supp_val = n_test*k_test
n_query = k_train*q_train

train(unet, optimizer, \
      train_mris_dl, val_mris_dl, \
      criterion1, criterion2, \
      n_supp_train, n_supp_val, \
      n_epochs, device)

print('Finished Training')

#query_plots(p, x_support, y_support)
