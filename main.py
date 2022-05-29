import torch
import torch.nn as nn

from torch.optim import Adam, SGD

from torch.utils.data import DataLoader

from models.encoder import FS_Encoder, NeuralNetwork, FS_Encoder2
from models.unet import UnetEncoder
from models.train import train

from datasets.dataloader import FewShot_Dataloader
from datasets.fewshot_sampler import NShotTaskSampler


evaluation_episodes = 1000
episodes_per_epoch = 7
n_epochs = 15

'''
k clases, n muestras, q queries
'''
n_train = 3 # n shots (train)
n_test = 1 # n shots (test)
k_train = 1 # k way (k classes) (train)
k_test = 1 # k way (k classes) (test)
q_train = 3 # q queries (train)
q_test = 5 # q queries (test)

print(f'Parametros: {episodes_per_epoch=}, {n_epochs=}, {n_train=}, {k_train=}, {q_train=}')

'''
Crear datasets
'''
#transform = transforms.Compose([transforms.ToTensor()])

train_mris = FewShot_Dataloader(
    '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS13DataNii/TrainingData', \
    'T1.nii', 
    'LabelsForTraining.nii', 
    48, 
    'training'
)

validating_mris = FewShot_Dataloader(
    '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS13DataNii/TrainingData', \
    'T1.nii', 
    'LabelsForTraining.nii', 
    48, 
    'validating'
)

train_mris_dl = DataLoader(
    train_mris, 
    batch_sampler=NShotTaskSampler(train_mris, episodes_per_epoch, n_train, k_train, q_train),
    num_workers=4
)

validating_mris_dl = DataLoader(
    validating_mris, 
    batch_sampler=NShotTaskSampler(validating_mris, episodes_per_epoch, n_test, k_test, q_test),
    num_workers=4
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# CUDA_LAUNCH_BLOCKING=1

unet = UnetEncoder(1, depth=5).to(device, dtype=torch.double)

# criterion1 = nn.CrossEntropyLoss()
# criterion2 = nn.CrossEntropyLoss()

criterion1 = nn.BCELoss()
criterion2 = nn.BCELoss()
#optimizer = SGD(unet.parameters(), lr=0.001, momentum=0.9)
optimizer = Adam(unet.parameters(), lr=0.0001)

n_support = n_train*k_train
n_query = k_train*q_train

train(unet, optimizer, train_mris_dl, criterion1, criterion2, n_support, n_epochs, device)

print('Finished Training')

#query_plots(p, x_support, y_support)
