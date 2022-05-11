from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.encoder import FS_Encoder, NeuralNetwork, FS_Encoder2
from models.metrics import euclidean_dist, cosine_dist, probs
from models.unet import UnetEncoder

from datasets.dataloader import FewShot_Dataloader
from datasets.fewshot_sampler import NShotTaskSampler

from utils.plots import plot_batch, query_plots

import matplotlib.pyplot as plt


evaluation_episodes = 1000
episodes_per_epoch = 1#00

'''
k clases, n muestras, q queries
'''

n_train = 5 # n shots (train)
n_test = 1 # n shots (test)
k_train = 1 # k way (k classes) (train)
k_test = 1 # k way (k classes) (test)
q_train = 5 # q queries (train)
q_test = 5 # q queries (test)

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
    #num_workers=4
)

validating_mris_dl = DataLoader(
    validating_mris, 
    batch_sampler=NShotTaskSampler(validating_mris, episodes_per_epoch, n_test, k_test, q_test),
    #num_workers=4
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

unet = UnetEncoder(1, depth=5).to(device, dtype=torch.double)

n_support = n_train*k_train
n_query = k_train*q_train

for data in tqdm(train_mris_dl):

    inputs, labels = data
    inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)

    x_support = inputs[:n_support]
    x_query = inputs[n_support:]

    y_support = labels[:n_support].to(device)
    # print(f'y unique = {torch.unique(y_support)}')
    y_query = labels[n_support:].to(device)

    plot_batch(x_support, y_support)

    outputs = unet(inputs)

    #print(f'outputs {outputs.shape}')

    f_support = outputs[:n_support]
    f_query = outputs[n_support:]

    idx_b = y_support==0 # Pertenecen al background
    idx_1 = y_support==1 # Pertenecen a la clase
    idx_b = idx_b.unsqueeze(axis=1)
    idx_1 = idx_1.unsqueeze(axis=1)

    f_support_masked = f_support*idx_1
    pr1 = torch.sum(f_support_masked, dim=(0, 2, 3))
    pr1 = pr1 / torch.sum(idx_1)
    print(f'pr1 shape = {pr1.view(1, pr1.shape[0], 1, 1).shape}')

    idx_q = y_query==1
    f_query_masked = f_query*idx_q.unsqueeze(axis=1)
    #prq = torch.sum(f_query_masked, dim=(2, 3))
    print(f'f_query_masked shape = {f_query_masked.shape}')

    # dists = euclidean_dist(f_query, prototype)
    dists = cosine_dist(pr1.view(1, pr1.shape[0], 1, 1), f_query_masked)#prq)

    p = probs(dists)

    # log_p_y = F.log_softmax(-dists, dim=1).view(k_train, q_train, -1)

# print(f'Size of x_support set = {x_support.shape}, Size of y_support set = {y_support.dtype} \n')
# print(f'Size of f_support set = {f_support.shape}, Size of f_query set = {f_query.shape} \n')
print(f'Size of prototype = {pr1.shape}\n')
print(f'Distancias = {dists.shape} \n')
print(f'Probabilities = {p.shape} \n')


print('Finished Training')

query_plots(p, x_query, y_query)
