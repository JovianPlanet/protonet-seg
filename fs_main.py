from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, lr_scheduler
#import torch.optim as optim

from torch.utils.data import DataLoader

from models.unet import UnetEncoder
from models.metrics import *

from datasets.dataloader import FewShot_Dataloader, UnetDataloader
from datasets.fewshot_sampler import NShotTaskSampler

from utils.plots import * 

torch.cuda.empty_cache()

evaluation_episodes = 1000
episodes_per_epoch = 5
n_epochs = 25
lr = 0.0001

train_heads = 8
val_heads = 2

TRAIN_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/train'
VAL_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/val'

'''
k clases, n muestras, q queries
'''
n_train = 12#3 # n shots (train)
k_train = 1 # k way (k classes) (train)
q_train = 12 # q queries (train)

n_val = 10#5 # n shots (val)
k_val = 1 # k way (k classes) (val)
q_val = 6#2 # q queries (val)

classes_dic = {'BGR': 0, 'GM': 1, 'WM': 2, 'CSF': 3}
batch_dice = {'GM': None, 'WM': None, 'CSF': None}
gen_dice = {'GM': 0.0, 'WM': 0.0, 'CSF': 0.0}

tdice = 0.

classes = ['GM', 'WM', 'CSF']
num_classes = len(classes)

cr = 'cross'

# lrs = [0.0001, 0.0003]

# for lr in lrs:

for heads in range(train_heads, 7, -1):

    PATH = './models/fsmul_wts-'+cr+'-h'+str(heads)+'-ep'+str(n_epochs)+'-10_.pth'

    print(f'\n\nParametros: {heads=}, \
                        {episodes_per_epoch=}, \
                        {n_epochs=}, \
                        {n_train=}, \
                        {k_train=}, \
                        {q_train=}, \
                        {lr=}, \
                        filename={PATH[9:]}\n'
    )

    '''
    Crear datasets
    '''
    # train_mris = FewShot_Dataloader(
    #     TRAIN_PATH,
    #     'T1.nii', 
    #     'LabelsForTraining.nii', 
    #     48, 
    #     heads
    # )

    train_mris = UnetDataloader(
        TRAIN_PATH,
        'T1.nii', 
        'LabelsForTraining.nii', 
        48, 
        heads
    )

    # val_mris = FewShot_Dataloader(
    #     VAL_PATH, 
    #     'T1.nii', 
    #     'LabelsForTraining.nii', 
    #     48, 
    #     val_heads
    # )

    val_mris = UnetDataloader(
        VAL_PATH, 
        'T1.nii', 
        'LabelsForTraining.nii', 
        48, 
        val_heads
    )

    # train_mris_dl = DataLoader(
    #     train_mris, 
    #     batch_sampler=NShotTaskSampler(train_mris, 
    #                                    episodes_per_epoch, 
    #                                    n_train, 
    #                                    k_train, 
    #                                    q_train, 
    #                                    fixed_tasks=[classes]
    #                                    ),
    # )

    train_mris_dl = DataLoader(
        train_mris, 
        batch_size=n_train+q_train,
        shuffle=True,
    )

    # val_mris_dl = DataLoader(
    #     val_mris, 
    #     batch_sampler=NShotTaskSampler(val_mris, 
    #                                    episodes_per_epoch, 
    #                                    n_val, 
    #                                    k_val, 
    #                                    q_val, 
    #                                    fixed_tasks=[classes]
    #                                    ),
    # )

    val_mris_dl = DataLoader(
        val_mris, 
        batch_size=n_val+q_val,
    )

    print(f'Tamano del dataset: {train_mris.df.shape[0]} slices \n')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    unet = UnetEncoder(1, depth=5, start_filts=32).to(device, dtype=torch.double)

    if cr == 'cross':

        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss()

    elif cr == 'dice':

        criterion1 = DiceLoss()
        criterion2 = DiceLoss()

    optimizer = Adam(unet.parameters(), lr=lr)

    #scheduler = lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 
                                               'min', 
                                               factor=0.5, 
                                               patience=10, 
                                               threshold=0.05,
                                               verbose=True
    )

    n_supp_train = n_train#len(classes)*n_train #n_train*k_train
    n_supp_val = n_val#len(classes)*n_val #n_val*k_val

    loss = 0.0
    
    best_dice = 0.0
    best_loss = 0.0

    for epoch_ in tqdm(range(n_epochs)):

        torch.cuda.empty_cache()

        running_loss = 0.0
        epoch_loss = 0.0
        epoch_dice = 0.0

        print(f'\n\nEpoch {epoch_ + 1}\n')
        
        unet.train()

        for i, episode in enumerate(train_mris_dl):

            torch.cuda.empty_cache()

            # loss1 = 0.0
            # loss2 = 0.0

            inputs, labels = episode
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)

            optimizer.zero_grad()

            y_support = labels[:n_supp_train].to(device)
            y_query = labels[n_supp_train:].to(device)

            #plot_batch(inputs[:n_supp_train], y_support)

            outputs = unet(inputs)

            f_support = outputs[:n_supp_train]
            f_query = outputs[n_supp_train:]

            del inputs, outputs

            pr1 = get_class_prototypes(f_support, y_support)

            dists1 = get_dists(f_query, pr1)
            m1 = probs(dists1)
            m1 = torch.argmax(m1, dim=1)

            if cr == 'dice':

                ''' Si criterion es Dice coefficient'''
                m1_ = F.one_hot(m1)
                m1_ = torch.autograd.Variable(m1_.double(), requires_grad=True)[:, :, :, :]
                y1 = F.one_hot(y_query.long()).double()[:, :, :, :]
                loss1 = criterion1(m1_[...,:], y1[...,:])
                print(f'{loss1=}')

                # for k1, v1 in classes_dic.items():

                #     loss1 += criterion1(m1[..., v1], y1[..., v1])

                # print(f'{loss1/4=}')

            elif cr == 'cross':

                '''Costo para cross-entropy'''
                loss1 = criterion1(dists1, y_query.long())

            del pr1, dists1 

            pr2 = get_class_prototypes(f_query, m1) #y_query)

            dists2 = get_dists(f_support, pr2)
            m2 = probs(dists2)
            m2 = torch.argmax(dists2, dim=1)

            if cr == 'dice':

                ''' Si el costo es Dice coefficient '''
                
                m2_ = F.one_hot(m2)
                m2_ = torch.autograd.Variable(m2_.double(), requires_grad=True)
                y2 = F.one_hot(y_support.long()).double()
                loss2 = criterion2(m2_[...,:], y2[...,:])

                #     x2 = m2==classes_dic[c2]
                #     m2_ = (m2*x2)/classes_dic[c2]
                #     y2 = y_support==classes_dic[c2]

                #     for im2 in range(x2.shape[0]):

                #         loss2 += criterion2(m2_[im2, :, :], ((y_support*y2)/classes_dic[c2])[im2, :, :])

            elif cr == 'cross':

                '''Costo para cross-entropy'''
                loss2 = criterion2(dists2, y_support.long())

            del y_support, y_query, f_support, f_query, pr2, dists2

            loss = loss1 + loss2
            running_loss += loss.item() 
            loss.backward()

            optimizer.step()

        epoch_loss = running_loss/(2*(i + 1)) # 4=numero clases + background 

        tdice = 0.0
        gen_dice = {'GM': 0.0, 'WM': 0.0, 'CSF': 0.0}

        unet.eval()
        with torch.no_grad():
            for ind, data in enumerate(val_mris_dl):
                
                x, y = data
                x = x.unsqueeze(1).to(device, dtype=torch.double)
                y = y.to(device, dtype=torch.double)

                x_q = x[n_supp_val:]

                y_s = y[:n_supp_val].to(device)
                y_q = y[n_supp_val:].to(device)

                f = unet(x)

                f_s = f[:n_supp_val]
                f_q = f[n_supp_val:]

                dice = 0.0

                pv = get_class_prototypes(f_s, y_s)

                dv = get_dists(f_q, pv)
                mv = torch.argmax(dv, dim=1)

                #plot_batch(mv, y_q)

                test = True
                if test:
                    t = F.one_hot(mv)
                    t = torch.autograd.Variable(t.double(), requires_grad=True)

                    tdice += dice_coeff(t[..., 1:], F.one_hot(y_q.long()).double()[..., 1:])

                for c in classes:

                    batch_dice[c] = dice_coeff(torch.where(mv==classes_dic[c], 1, 0), 
                                      torch.where(y_q==classes_dic[c], 1, 0)
                    )
                    
                    gen_dice[c] += batch_dice[c].item()

                    #plot_batch_full(x_q, y_q, mv)

                #print(f'tdice = {tdice}, gen = {(gen_dice["GM"]+gen_dice["WM"])/2}')

        gen_dice = {k: v / (ind+1) for k, v in gen_dice.items()}
        epoch_dice = sum(gen_dice.values())/num_classes

        if epoch_ == 0: 
            best_loss = epoch_loss
            best_dice = epoch_dice

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        print(f'Loss = {epoch_loss:.3f}, Best loss = {best_loss:.3f}\n')

        for key, value in gen_dice.items():
            print(f'Validation {key} dice = {value:.3f}')
        print(f'one hot dice = {tdice/(ind+1)}')

        if epoch_dice > best_dice:
            best_dice = epoch_dice
            #print(f'\nUpdated weights file!')
            torch.save(unet.state_dict(), PATH)
            
        print(f'\nDice = {epoch_dice:.3f}, Best dice = {best_dice:.3f}\n')

        #print(f'lr = {scheduler.get_last_lr()}\n')

        #scheduler.step(epoch_loss)

print('Finished Training')

