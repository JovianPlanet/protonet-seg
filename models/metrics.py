import torch
from torch import nn

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
    # print(f'shape distancias = {cos(x, y).shape}')
        
    return cos(x, y) #cos(x.unsqueeze(0), y)

def probs(x):
    p = nn.Softmax(dim=0) #ReLU()

    return  p(x) #torch.sigmoid(x)


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
        #print(f'{inputs.unique()=}, {targets.unique()=}')
        
        intersection = (inputs * targets).sum()                            
        dice = torch.mean((2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth))  
        
        return 1.0 - dice