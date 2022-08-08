from tqdm import tqdm
import torch
import torch.optim as optim

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
          train_heads,
          device):

    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)

    '''

    'GM' : 1, 
    'BG' : 2, 
    'WM' : 3, 
    'WML' : 4,
    'CSF' : 5, 
    'VEN' : 6, 
    'CER' : 7, 
    'BSTEM' : 8,
    'BGR' : 0

    '''

    classes = ['GM', 'WM', 'CSF']
    num_classes = len(classes)

    cr = 'dice'

    PATH = './models/fsmul_wts-'+cr+'-h'+str(train_heads)+'-ep'+str(epoch_size)+'-10.pth'

    loss = 0.0
    best_dice = 0.0
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

            loss1 = 0.0
            loss2 = 0.0

            torch.cuda.empty_cache()

            inputs, labels = episode
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)

            optimizer.zero_grad()

            y_support = labels[:n_supp_train].to(device)
            y_query = labels[n_supp_train:].to(device)

            # plot_batch(inputs[:n_supp_train], y_support)

            outputs = model(inputs)

            f_support = outputs[:n_supp_train]
            f_query = outputs[n_supp_train:]

            del inputs, outputs

            n_train = n_supp_train//num_classes
            q_train = y_query.shape[0]//num_classes

            d1 = get_prototype_all(f_support, y_support, f_query, n_train, q_train)

            if cr == 'dice':

                ''' Si criterion es Dice coefficient'''
                for j, d in enumerate(d1):

                    x1 = d>0.9
                    y1 = torch.where(y_query[j*q_train:(j*q_train)+q_train,:,:].double()>0.0, 1.0, 0.0)

                    loss1 += criterion1(d, y1)

            elif cr == 'cross':

                '''Costo para cross-entropy'''
                p1 = torch.flatten(torch.stack(d1, dim=1), start_dim=2, end_dim=3)
                loss1 = criterion1(p1, torch.flatten(y_query, start_dim=1, end_dim=2).long())

            del d1 

            d2 = get_prototype_all(f_query, y_query, f_support, q_train, n_train)

            if cr == 'dice':

                ''' Si el costo es Dice coefficient '''
                for m, d_ in enumerate(d2):

                    x2 = d_>0.9
                    y2 = torch.where(y_support[m*n_train:(m*n_train)+n_train,:,:].double()>0.0, 1.0, 0.0)

                    loss2 += criterion2(d_, y2)

            elif cr == 'cross':

                '''Costo para cross-entropy'''
                p2 = torch.flatten(torch.stack(d2, dim=1), start_dim=2, end_dim=3)
                loss2 = criterion2(p2, torch.flatten(y_support, start_dim=1, end_dim=2).long())

            del y_support, y_query, f_support, f_query, d2

            loss = loss1 + loss2
            running_loss += loss.item() 
            loss.backward()

            optimizer.step()

        epoch_loss = running_loss/(num_classes*(i + 1)) # 4=numero clases + background 

        model.eval()
        with torch.no_grad():
            for ind, data in enumerate(val_mri):
                
                x, y = data
                x = x.unsqueeze(1).to(device, dtype=torch.double)
                y = y.to(device, dtype=torch.double)

                x_q = x[n_supp_val:]

                y_s = y[:n_supp_val].to(device)
                y_q = y[n_supp_val:].to(device)

                f = model(x)

                f_s = f[:n_supp_val]
                f_q = f[n_supp_val:]

                dice = 0.0
                n_val = n_supp_val//num_classes
                q_val = y_q.shape[0]//num_classes

                dv = get_prototype_all(f_s, y_s, f_q, n_val, q_val)

                for k, d in enumerate(dv):

                    y_q_ = torch.where(y_q[k*q_val:(k*q_val)+q_val,:,:].double()>0.0, 1.0, 0.0)

                    dice += dice_coeff(d>0.9, y_q_)
                    print(f'{dice_coeff(d, y_q_)}, {dice_coeff(d>0.9, y_q_)}, {nn.CosineSimilarity(dim=0)((d>0.9).view(-1), y_q_.view(-1))}')

                    # plot_batch_full(x_q.squeeze(1)[k*q_val:(k*q_val)+q_val,:,:], 
                    #                 y_q_, 
                    #                 d>0.9
                    # )

                running_dice += dice/num_classes

        epoch_dice = running_dice/(ind + 1)
        if epoch_ == 0: 
            best_loss = epoch_loss
            best_dice = epoch_dice

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        print(f'Loss = {epoch_loss:.3f}, Best loss = {best_loss:.3f}')

        if epoch_dice > best_dice:
            best_dice = epoch_dice
            print(f'Updated weights file!')
            torch.save(model.state_dict(), PATH)
            
        print(f'Dice = {epoch_dice:.3f}, Best dice = {best_dice:.3f}\n')

        #print(f'{scheduler.get_last_lr()=}')

        scheduler.step()
