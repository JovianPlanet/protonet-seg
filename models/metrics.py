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


# class DiceLossMultiClass(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#         self.label_dict = { 'GM' : 1, 'BG' : 2, 'WM' : 3, 'WML' : 4,
#                             'CSF' : 5, 'VEN' : 6, 'CER' : 7, 'BSTEM' : 8}

#         self.classes = ['GM', 'WM', 'CSF']

#     def coef(self, x, y, smooth=1e-6):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         #x = F.sigmoid(x)       
        
#         #flatten label and prediction tensors
#         x = x.view(-1)
#         y = y.view(-1)
        
#         intersection = (x * y).sum()                            
#         dice = torch.mean((2.*intersection + smooth)/(x.sum() + y.sum() + smooth))  
        
#         return dice

#     def forward(self, inputs, targets, numLabels):
#         d = 0
#         for i in range(numLabels):
#             dice += self.coef(inputs)


# def dice_coef(y_true, y_pred):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.sum(y_true_f * y_pred_f)
#     smooth = 0.0001
#     return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# def dice_coef_multilabel(y_true, y_pred, numLabels):
#     dice=0
#     for index in range(numLabels):
#         dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
#     return dice/numLabels # taking average

# num_class = 5

# imgA = np.random.randint(low=0, high= 2, size=(5, 64, 64, num_class) ) # 5 images in batch, 64 by 64, num_classes map
# imgB = np.random.randint(low=0, high= 2, size=(5, 64, 64, num_class) )


# plt.imshow(imgA[0,:,:,0]) # for 0th image, class 0 map
# plt.show()

# plt.imshow(imgB[0,:,:,0]) # for 0th image, class 0 map
# plt.show()

# dice_score = dice_coef_multilabel(imgA, imgB, num_class)
# print(f'For A and B {dice_score}')

# dice_score = dice_coef_multilabel(imgA, imgA, num_class)
# print(f'For A and A {dice_score}')