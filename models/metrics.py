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
    p = nn.Softmax(dim=0)
    return p(x)

