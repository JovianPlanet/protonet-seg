from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.encoder import FS_Encoder, NeuralNetwork, FS_Encoder2
from datasets.dataloader import FewShot_Dataloader
from datasets.fewshot_sampler import NShotTaskSampler
from models.metrics import euclidean_dist


evaluation_episodes = 1000
episodes_per_epoch = 100

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

enc = FS_Encoder().to(device, dtype=torch.double)

nnet = NeuralNetwork().to(device, dtype=torch.double)

enc2 = FS_Encoder2().to(device, dtype=torch.double)

n_support = n_train*k_train
n_query = k_train*q_train

for data in tqdm(train_mris_dl):

    inputs, labels = data
    inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)
    #print(f'inputs {inputs.dtype}')

    outputs = enc(inputs)
    #print(f'outputs {outputs.shape}')

    support = outputs[:n_support]
    query = outputs[n_support:]
    # print(f'Size of support set = {support.shape}, Size of query set = {query.shape} \n')

    prototype = torch.sum(support, dim=0, keepdims=True)/support.shape[1]
    # print(f'Size of prototype = {prototype.shape}\n')

    dists = euclidean_dist(query, prototype)

    log_p_y = F.log_softmax(-dists, dim=1).view(k_train, q_train, -1)

print(f'Size of prototype = {prototype.shape}\n')
print(f'Distancias = {dists} \n')
print(f'Probabilities = {log_p_y} \n')


print('Finished Training')