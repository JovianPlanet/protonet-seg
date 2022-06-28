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

torch.cuda.empty_cache()

evaluation_episodes = 1000
episodes_per_epoch = 10
n_epochs = 150
lr = 0.0001

TRAIN_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/train'
VAL_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/val'

'''
k clases, n muestras, q queries
'''
n_train = 3 # n shots (train)
k_train = 1    # k way (k classes) (train)
q_train = 3 # q queries (train)

n_val = 5 # n shots (val)
k_val = 1 # k way (k classes) (val)
q_val = 1 # q queries (val)

classes = ['GM', 'WM', 'CSF']

print(f'Parametros: {episodes_per_epoch=}, \
                    {n_epochs=}, \
                    {n_train=}, \
                    {k_train=}, \
                    {q_train=}, \
                    {lr=}'
)

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
    batch_sampler=NShotTaskSampler(train_mris, 
                                   episodes_per_epoch, 
                                   n_train, 
                                   k_train, 
                                   q_train, 
                                   fixed_tasks=[classes]
                                   ),
)

val_mris_dl = DataLoader(
    val_mris, 
    batch_sampler=NShotTaskSampler(val_mris, 
                                   episodes_per_epoch, 
                                   n_val, 
                                   k_val, 
                                   q_val, 
                                   fixed_tasks=[classes]
                                   ),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

unet = UnetEncoder(1, depth=5).to(device, dtype=torch.double)

# criterion1 = nn.CrossEntropyLoss()
# criterion2 = nn.CrossEntropyLoss()

criterion1 = DiceLoss()
criterion2 = DiceLoss()

optimizer = Adam(unet.parameters(), lr=lr)

n_supp_train = len(classes)*n_train #n_train*k_train
n_supp_val = len(classes)*n_val #n_val*k_val
n_query = k_train*q_train

train(unet, 
      optimizer,
      train_mris_dl, 
      val_mris_dl,
      criterion1, 
      criterion2,
      n_supp_train, 
      n_supp_val,
      n_epochs, 
      device
)

print('Finished Training')

#query_plots(p, x_support, y_support)
