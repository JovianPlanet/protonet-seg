import torch
from torch import nn

def get_prototype(f, y, class_):

    classes = { 'GM' : 1, 
                'WM' : 2, 
                'CSF' : 3, 
                'BGR' : 0
    }

    idx = y==classes[class_] # Pertenecen a la clase
    idx = idx.unsqueeze(axis=1)

    f_masked = f*idx

    p = torch.sum(f_masked, dim=(0, 2, 3))
    sum_ = torch.sum(idx)
    if sum_ == 0:
        p = p*0.0
    else:
        p = p / sum_

    return p.view(1, p.shape[0], 1, 1)

def get_prototype_all(f, y, f_q):

    classes = { 'BGR' : 0, 
                'GM' : 1, 
                'WM' : 2, 
                'CSF' : 3, 
    }

    dists = []
    for key, value in classes.items():

        idx = y==value # Pertenecen a la clase
        idx = idx.unsqueeze(axis=1)

        f_masked = f*idx

        p = torch.sum(f_masked, dim=(0, 2, 3))
        sum_ = torch.sum(idx)
        if sum_ == 0:
            p = p*0.0
        else:
            p = p / sum_

        d = cosine_dist(p.view(1, p.shape[0], 1, 1), f_q)
        dists.append(d)

    return dists

def euclidean_dist(x, y):

    """
    Computes euclidean distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cosine_dist(x, y):
    cos = nn.CosineSimilarity(dim=1)
        
    return cos(x, y)

def probs(x):
    p = nn.Softmax(dim=1) #ReLU()

    return p(x) #torch.sigmoid(x) # 

def dice_coeff(x, y, smooth=1e-6):

    #comment out if your model contains a sigmoid or equivalent activation layer
    #x = F.sigmoid(x)       
    
    #flatten label and prediction tensors
    x = x.view(-1)
    y = y.view(-1)
    
    intersection = (x * y).sum()                            
    dice = torch.mean((2.*intersection + smooth)/(x.sum() + y.sum() + smooth))  
    
    return dice

#PyTorch
#https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = torch.mean((2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth))  
        
        return 1.0 - dice
