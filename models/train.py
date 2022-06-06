from tqdm import tqdm
import torch
import torch.optim as optim
from torch.autograd import Variable

from models.metrics import *
from utils.plots import * 

def train(model, optimizer, \
          train_mris, val_mri, \
          criterion1, criterion2, \
          n_supp_train, n_supp_val, \
          epoch_size, device):

    PATH = './models/fs_weights.pth'
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5, last_epoch=-1)
    loss = 0.0
    best_score = 2.0

    for epoch_ in tqdm(range(epoch_size)):

        running_dice = 0.0
        running_loss = 0.0
        epoch_loss = 0.0

        print(f'\n\nEpoch {epoch_ + 1}\n')
        
        model.train()

        for i, episode in enumerate(train_mris):

            inputs, labels = episode
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)

            optimizer.zero_grad()

            y_support = labels[:n_supp_train].to(device)
            y_query = labels[n_supp_train:].to(device)

            #plot_batch(inputs, labels)

            outputs = model(inputs)
            assert not torch.isnan(outputs).any(), f'Hay outputs nan'

            f_support = outputs[:n_supp_train]
            f_query = outputs[n_supp_train:]

            idx_b = y_support==0 # Pertenecen al background
            idx_1 = y_support==1 # Pertenecen a la clase
            idx_b = idx_b.unsqueeze(axis=1)
            idx_1 = idx_1.unsqueeze(axis=1)

            f_support_masked = f_support*idx_1

            pr1 = torch.sum(f_support_masked, dim=(0, 2, 3))
            sum_ = torch.sum(idx_1)
            if sum_ == 0:
                pr1 = pr1*0.0
            else:
                pr1 = pr1 / sum_
            pr1 = pr1.view(1, pr1.shape[0], 1, 1)

            idx_q = y_query==1
            idx_q = idx_q.unsqueeze(axis=1)
            f_query_masked = f_query*idx_q

            dists1 = cosine_dist(pr1, f_query)#f_query_masked)
            assert not torch.isnan(dists1).any(), f'Distancias1 tiene valores nan'

            #p1 = probs(dists1)

            loss1 = criterion1(dists1, y_query.double())#.float())

            prq = torch.sum(f_query_masked, dim=(0,2,3))
            sum_ = torch.sum(idx_q)
            if sum_ == 0:
                prq = prq*0.0
            else:
                prq = prq / sum_
            prq = prq.view(1, prq.shape[0], 1, 1)

            dists2 = cosine_dist(prq, f_support)#f_support_masked)
            assert not torch.isnan(dists2).any(), f'Distancias2 tiene valores nan'

            #p2 = probs(dists2)

            loss2 = criterion2(dists2, y_support.double())#.float())
            loss = loss1 + loss2
            running_loss += loss.item() 
            loss.backward()

            #print(f'Loss = {loss.item():.3f}')

            optimizer.step()

        epoch_loss = running_loss/(i + 1)
        if epoch_loss < best_score:
            best_score = epoch_loss
            print(f'Updated weights file!')
            torch.save(model.state_dict(), PATH)
        print(f'loss = {epoch_loss:.3f}, {best_score=:.3f}')

        model.eval()
        with torch.no_grad():
            for j, data in enumerate(val_mri):
                
                x, y = data
                x = x.unsqueeze(1).to(device, dtype=torch.double)
                y = y.to(device, dtype=torch.double)

                y_s = y[:n_supp_val].to(device)
                y_q = y[n_supp_val:].to(device)

                f = model(x)

                f_s = f[:n_supp_val]
                f_q = f[n_supp_val:]

                idx_b = y_s==0 # Pertenecen al background
                idx_1 = y_s==1 # Pertenecen a la clase
                idx_b = idx_b.unsqueeze(axis=1)
                idx_1 = idx_1.unsqueeze(axis=1)

                f_s_mask = f_s*idx_1

                proto = torch.sum(f_s_mask, dim=(0, 2, 3))
                sum_ = torch.sum(idx_1)
                if sum_ == 0:
                    proto = proto*0.0
                else:
                    proto = proto / sum_
                proto = proto.view(1, proto.shape[0], 1, 1)

                dists = cosine_dist(proto, f_q)

                pval = torch.where(dists>0.8, 1, 0)
                dice = dice_coeff(pval, y_q)
                running_dice += dice
                #print(f'Val dice = {dice.item():.3f}')

        print(f'Val dice = {running_dice/(j + 1):.3f}\n')

        #scheduler.step()