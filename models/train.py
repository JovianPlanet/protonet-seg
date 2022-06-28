from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np

from models.metrics import *
from utils.plots import * 

def train(model, 
          optimizer,
          train_mris, 
          val_mri,
          criterion1, 
          criterion2,
          n_supp_train, 
          n_supp_val,
          epoch_size, 
          device):

    PATH = './models/fsmul_wts_ep'+epoch_size+'.pth'
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5, last_epoch=-1)

    classes_dic = {'GM' : 1, 
                   'BG' : 2, 
                   'WM' : 3, 
                   'WML' : 4,
                   'CSF' : 5, 
                   'VEN' : 6, 
                   'CER' : 7, 
                   'BSTEM' : 8,
                   'BGR' : 0
    }

    classes = ['GM', 'WM', 'CSF']
    num_classes = len(classes)

    loss = 0.0
    best_dice = 4.0
    best_loss = 0.0

    for epoch_ in tqdm(range(epoch_size)):

        torch.cuda.empty_cache()

        running_dice = 0.0
        running_loss = 0.0
        epoch_loss = 0.0
        epoch_dice = 0.0

        print(f'\n\nEpoch {epoch_ + 1}\n')
        
        model.train()

        for i, episode in enumerate(train_mris):

            torch.cuda.empty_cache()

            inputs, labels = episode
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)

            optimizer.zero_grad()

            y_support = labels[:n_supp_train].to(device)
            y_query = labels[n_supp_train:].to(device)

            #plot_batch(inputs[n_supp_train:], y_query)

            outputs = model(inputs)
            #assert not torch.isnan(outputs).any(), f'Hay outputs nan'

            f_support = outputs[:n_supp_train]
            f_query = outputs[n_supp_train:]

            del inputs, outputs

            d1 = get_prototype_all(f_support, y_support, f_query)

            loss1 = criterion1(d1[0], y_query.double())

            for j, d in enumerate(d1[1:]):

                loss1 += criterion1(d[j*num_classes:(j*num_classes)+num_classes,:,:], 
                              y_query[j*num_classes:(j*num_classes)+num_classes,:,:].double()
                )

            # Costo para cross-entropy
            # p1 = torch.flatten(torch.stack(d1, dim=1), start_dim=2, end_dim=3)
            # loss1 = criterion1(p1, torch.flatten(y_query, start_dim=1, end_dim=2).long())

            del d1 

            d2 = get_prototype_all(f_query, y_query, f_support)

            # Costo para cross-entropy
            # p2 = torch.flatten(torch.stack(d2, dim=1), start_dim=2, end_dim=3)
            # loss2 = criterion2(p2, torch.flatten(y_support, start_dim=1, end_dim=2).long())

            loss2 = criterion2(d2[0], y_support.double())

            for m, d3 in enumerate(d2[1:]):

                loss2 += criterion2(d3[m*num_classes:(m*num_classes)+num_classes,:,:], 
                            y_support[m*num_classes:(m*num_classes)+num_classes,:,:].double()
                )

            del y_support, y_query, f_support, f_query, d2, d3

            loss = loss1 + loss2
            running_loss += loss.item() 
            loss.backward()

            optimizer.step()

        epoch_loss = running_loss/(4*(i + 1)) # 4=numero clases + background
        if epoch_ == 0: 
            best_loss = epoch_loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            #print(f'Updated weights file!')
            #torch.save(model.state_dict(), PATH)
        print(f'loss = {epoch_loss:.3f}, {best_loss=:.3f}')

        model.eval()
        with torch.no_grad():
            for ind, data in enumerate(val_mri):
                
                x, y = data
                x = x.unsqueeze(1).to(device, dtype=torch.double)
                y = y.to(device, dtype=torch.double)

                y_s = y[:n_supp_val].to(device)
                y_q = y[n_supp_val:].to(device)

                f = model(x)

                f_s = f[:n_supp_val]
                f_q = f[n_supp_val:]

                dv = get_prototype_all(f_s, y_s, f_q)

                dice = 0.0
                n_val = n_supp_val//num_classes
                for k, d in enumerate(dv[1:]):

                    dice += dice_coeff(d[k*n_val:(k*n_val)+n_val,:,:], 
                                       y_q[k*n_val:(k*n_val)+n_val,:,:].double()
                    )

                    # plot_batch_full(x_q.squeeze(1)[k*n_val:(k*n_val)+n_val,:,:], 
                    #                 y_q[k*n_val:(k*n_val)+n_val,:,:], 
                    #                 d[k*n_val:(k*n_val)+n_val,:,:]>0.5
                    # )

                running_dice += dice/num_classes

        epoch_dice = running_dice/(ind + 1)
        if epoch_ == 0: 
            best_dice = epoch_dice
        if epoch_dice < best_dice:
            best_dice = epoch_dice
            print(f'Updated weights file!')
            torch.save(model.state_dict(), PATH)
        print(f'Val dice = {epoch_dice:.3f}, {best_dice=:.3f}\n')

        #scheduler.step()