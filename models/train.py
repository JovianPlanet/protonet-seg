from tqdm import tqdm
import torch
import torch.optim as optim
from torch.autograd import Variable

from models.metrics import euclidean_dist, cosine_dist, probs
from utils.plots import * #plot_batch, query_plots, 

def train(model, optimizer, train_mris_dl, criterion1, criterion2, n_support, epoch_size, device):

    #scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    loss = 0.0

    for epoch_ in tqdm(range(epoch_size)):

        for episode in train_mris_dl: #tqdm(train_mris_dl):
            dists1 = 0.0
            dists2 = 0.0

            inputs, labels = episode
            in_ = inputs<0
            #print(f'Total de pixel < 0 en imagen de entrada: {torch.sum(in_)}')
            #print(f'{inputs.argmax()=}, {inputs.argmin().item()=}')
            #print(f'{inputs.max()=}, {inputs.min().item()=}')
            #print(f'{torch.isnan(inputs).unique()=}')
            assert not torch.isnan(inputs).any(), f'Hay entradas nan'
            #assert not in_.any(), f'Hay entradas < 0'
            inputs[in_] = 0.0
            assert not (inputs<0).any(), f'Hay entradas < 0'
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)

            optimizer.zero_grad()

            #x_support = inputs[:n_support]
            #x_query = inputs[n_support:]

            y_support = labels[:n_support].to(device)
            # print(f'y unique = {torch.unique(y_support)}')
            y_query = labels[n_support:].to(device)

            #plot_batch(inputs, labels)

            outputs = model(inputs)
            assert not torch.isnan(outputs).any(), f'Hay outputs nan'

            f_support = outputs[:n_support]
            f_query = outputs[n_support:]

            idx_b = y_support==0 # Pertenecen al background
            idx_1 = y_support==1 # Pertenecen a la clase
            idx_b = idx_b.unsqueeze(axis=1)
            idx_1 = idx_1.unsqueeze(axis=1)

            f_support_masked = f_support*idx_1
            pr1 = torch.sum(f_support_masked, dim=(0, 2, 3))
            sum_ = torch.sum(idx_1)
            if sum_ == 0:
                pr1 = pr1*0.0
                #print(f'shape de pr1 cuando no hay clase: {pr1.shape}')
            else:
                pr1 = pr1 / sum_
            pr1 = pr1.view(1, pr1.shape[0], 1, 1)

            assert not torch.isnan(pr1).any(), f'nan en pr1. \
            Sum idx_1: {torch.sum(idx_1)}, \
            Cantidad en nans en pr1: {torch.sum(torch.isnan(pr1))}'

            idx_q = y_query==1
            idx_q = idx_q.unsqueeze(axis=1)
            f_query_masked = f_query*idx_q


            dists1 = cosine_dist(pr1, f_query_masked)#prq)
            #print(f'{dists1.max().item()=:.2f}, {dists1.min().item()=}')
            #print(f'{dists1.argmax()=}, {dists1.argmin().item()=}')
                
            assert not torch.isnan(dists1).any(), f'Distancias tiene valores nan'
            #dists1 = torch.nan_to_num(dists1) #torch.where(dists1==torch.nan, 0.0, dists1)
            #print(f'{dists1.max().item()=}, {dists1.min().item()=}\n')
            

            p1 = probs(dists1)

            #m1 = torch.where(p1>0.25, 1, 0).double()
            #m1 = Variable(m1.data, requires_grad=True)

            #gen_masks = torch.where(p>0.25, 1, 0).short()
            #idx_gen = gen_masks==1
            #idx_gen = idx_gen.unsqueeze(axis=1)

            loss = criterion1(p1, y_query.double())#.float())

            prq = torch.sum(f_query_masked, dim=(0,2,3))
            prq = prq / torch.sum(idx_q)
            prq = prq.view(1, prq.shape[0], 1, 1)

            dists2 = cosine_dist(prq, f_support_masked)
            #dists2 = torch.cdist(prq, f_support_masked, p=2.0)
            dists2 = torch.nan_to_num(dists2)

            p2 = probs(dists2)

            #m2 = torch.where(p2>p2.max()/2, 1, 0).double()
            #m2 = Variable(m2.data, requires_grad=True)

            loss += criterion2(p2, y_support.double())#.float())
            loss.backward()

            #print(f'Loss interna = {loss.item()}\n')

            optimizer.step()

        #scheduler.step()
        print(f'Loss = {loss.item()}\n')

    #torch.save(model.state_dict(), PATH)