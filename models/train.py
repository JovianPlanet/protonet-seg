from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np

from models.metrics import *
from utils.plots import * 

def train(model, optimizer, \
          train_mris, val_mri, \
          criterion1, criterion2, \
          n_supp_train, n_supp_val, \
          epoch_size, device):

    PATH = './models/fsmulti_weights.pth'
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5, last_epoch=-1)
    classes = { 'GM' : 1, 'BG' : 2, 
                'WM' : 3, 'WML' : 4,
                'CSF' : 5, 'VEN' : 6, 
                'CER' : 7, 'BSTEM' : 8,
                'BGR' : 0
    }
    loss = 0.0
    best_score = 2.0

    for epoch_ in tqdm(range(epoch_size)):

        torch.cuda.empty_cache()

        running_dice = 0.0
        running_loss = 0.0
        epoch_loss = 0.0

        print(f'\n\nEpoch {epoch_ + 1}\n')
        
        model.train()

        for i, episode in enumerate(train_mris):

            torch.cuda.empty_cache()

            inputs, labels = episode
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)

            optimizer.zero_grad()

            y_support = labels[:n_supp_train].to(device)
            y_query = labels[n_supp_train:].to(device)

            #print(f'{y_query.unique()=}')

            #plot_batch(inputs, labels)

            outputs = model(inputs)
            #assert not torch.isnan(outputs).any(), f'Hay outputs nan'

            f_support = outputs[:n_supp_train]
            f_query = outputs[n_supp_train:]

            del inputs, outputs

            d1 = get_prototype_all(f_support, y_support, f_query)

            p1 = probs(torch.stack(d1, dim=1))
            p1 = torch.flatten(torch.stack(d1, dim=1), start_dim=2, end_dim=3)

            # print(f'{torch.stack(d1, dim=1).shape=}')
            # print(f'{p1.shape=}')
            # print(f'{y_query.unique()=}')
            # print(f'{y_query.shape=}')

            loss1 = criterion1(p1, torch.flatten(y_query, start_dim=1, end_dim=2).long())#y_query.double())

            del d1, p1

            d2 = get_prototype_all(f_query, y_query, f_support)

            p2 = probs(torch.stack(d2, dim=1))
            p2 = torch.flatten(torch.stack(d2, dim=1), start_dim=2, end_dim=3)

            loss2 = criterion2(p2, torch.flatten(y_support, start_dim=1, end_dim=2).long())

            del y_support, y_query, f_support, f_query, d2, p2

            # loss2 = criterion2(dq1, y_support.double())#.float())
            loss = loss1 + loss2
            running_loss += loss.item() 
            loss.backward()

            optimizer.step()

        epoch_loss = running_loss/(i + 1)
        if epoch_ == 0: 
            best_score = epoch_loss
        if epoch_loss < best_score:
            best_score = epoch_loss
            print(f'Updated weights file!')
            torch.save(model.state_dict(), PATH)
        print(f'loss = {epoch_loss:.3f}, {best_score=:.3f}')

        # model.eval()
        # with torch.no_grad():
        #     for j, data in enumerate(val_mri):
                
        #         x, y = data
        #         x = x.unsqueeze(1).to(device, dtype=torch.double)
        #         y = y.to(device, dtype=torch.double)

        #         y_s = y[:n_supp_val].to(device)
        #         y_q = y[n_supp_val:].to(device)

        #         f = model(x)

        #         f_s = f[:n_supp_val]
        #         f_q = f[n_supp_val:]

        #         idx_b = y_s==0 # Pertenecen al background
        #         idx_1 = y_s==1 # Pertenecen a la clase
        #         idx_b = idx_b.unsqueeze(axis=1)
        #         idx_1 = idx_1.unsqueeze(axis=1)

        #         f_s_mask = f_s*idx_1

        #         proto = torch.sum(f_s_mask, dim=(0, 2, 3))
        #         sum_ = torch.sum(idx_1)
        #         if sum_ == 0:
        #             proto = proto*0.0
        #         else:
        #             proto = proto / sum_
        #         proto = proto.view(1, proto.shape[0], 1, 1)

        #         dists = cosine_dist(proto, f_q)

        #         pval = torch.where(dists>0.8, 1, 0)
        #         dice = dice_coeff(pval, y_q)
        #         running_dice += dice
        #         #print(f'Val dice = {dice.item():.3f}')

        # print(f'Val dice = {running_dice/(j + 1):.3f}\n')

        #scheduler.step()